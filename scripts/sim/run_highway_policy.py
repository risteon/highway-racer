#! /usr/bin/env python
import warnings
from typing import Union, Tuple
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from moviepy.editor import ImageSequenceClip
from tqdm import trange

import gymnasium as gym
import highway_env  # This registers highway environments
from gymnasium.wrappers import TimeLimit, FlattenObservation

import jax
from ml_collections import config_flags
from flax.training.checkpoints import restore_checkpoint
from absl import app, flags

from jaxrl5.agents import (
    SACLearner,
    DistributionalSACLearner,
)

jax.config.update("jax_platform_name", "cpu")

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

flags.DEFINE_string("policy_file", None, "Path to the policy checkpoint directory")
flags.DEFINE_string("video_output_dir", "./evaluation_videos", "Path to the video output directory")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes")
flags.DEFINE_integer("max_steps", 1000, "Maximum steps per episode")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_boolean("render", True, "Enable video rendering")
flags.DEFINE_boolean("save_metrics", True, "Save detailed metrics to JSON file")
flags.DEFINE_string("env_name", "highway-v0", "Highway environment name")

config_flags.DEFINE_config_file(
    "config",
    "configs/highway_distributional_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def safety_reward_fn(obs, info=None):
    """
    Collision risk-based safety reward for highway driving.
    Penalizes close proximity to other vehicles.
    
    Args:
        obs: Flattened highway environment observation (90,) = 15 vehicles × 6 features
        info: Additional environment info (optional)
    
    Returns:
        safety_reward: Negative reward for collision risk
    """
    # Reshape flattened observation back to (15, 6)
    obs = obs.reshape(15, 6)  # [presence, x, y, vx, vy, heading]
    
    # Filter for present vehicles (presence > 0.5)
    present_vehicles = obs[obs[:, 0] > 0.5]
    
    if len(present_vehicles) <= 1:
        return 0.0  # Only ego vehicle or no vehicles, no collision risk
    
    ego_vehicle = present_vehicles[0]  # First present vehicle is ego
    other_vehicles = present_vehicles[1:]  # Rest are other vehicles
    
    # Compute distances to all other vehicles
    ego_pos = ego_vehicle[1:3]  # [x, y] (skip presence feature)
    other_pos = other_vehicles[:, 1:3]  # [n_vehicles, 2]
    
    distances = np.linalg.norm(other_pos - ego_pos, axis=1)
    min_distance = np.min(distances)
    
    # Safety parameters
    safety_threshold = 10.0  # meters - safe following distance
    collision_threshold = 2.0  # meters - collision imminent
    
    if min_distance < collision_threshold:
        collision_risk = -1.0  # High penalty for imminent collision
    elif min_distance < safety_threshold:
        # Exponential penalty for unsafe proximity
        collision_risk = -np.exp(-(min_distance - collision_threshold) / 
                                (safety_threshold - collision_threshold))
    else:
        collision_risk = 0.0  # Safe distance
    
    return collision_risk


def load_highway_agent(agent: Union[SACLearner, DistributionalSACLearner], policy_file: str):
    """
    Load the highway agent from a checkpoint.
    """
    param_dict = {
        "actor": agent.actor,
        "critic": agent.critic,
        "target_critic_params": agent.target_critic,
        "temp": agent.temp,
        "rng": agent.rng,
    }
    
    # Add optional components for DistributionalSACLearner
    if hasattr(agent, "limits"):
        param_dict["limits"] = agent.limits
    if hasattr(agent, "q_entropy_lagrange"):
        param_dict["q_entropy_lagrange"] = agent.q_entropy_lagrange
    
    param_dict = restore_checkpoint(policy_file, target=param_dict)
    
    # Create replacement dict
    replace_dict = {
        "actor": param_dict["actor"],
        "critic": param_dict["critic"],
        "target_critic": param_dict["target_critic_params"],
        "temp": param_dict["temp"],
        "rng": param_dict["rng"],
    }
    
    # Add optional components if they exist
    if "limits" in param_dict:
        replace_dict["limits"] = param_dict["limits"]
    if "q_entropy_lagrange" in param_dict:
        replace_dict["q_entropy_lagrange"] = param_dict["q_entropy_lagrange"]
    
    return agent.replace(**replace_dict)


def run_highway_trajectory(agent, env: gym.Env, max_steps=1000, render_video=False):
    """
    Run a single highway trajectory with the agent and collect metrics.
    """
    obs, _ = env.reset()
    if hasattr(agent, "env_reset"):
        agent = agent.env_reset(obs)
    
    images = []
    episode_return = 0
    episode_length = 0
    safety_violations = 0
    min_distances = []
    ego_speeds = []
    collision_occurred = False
    
    for step in range(max_steps):
        if render_video:
            img = env.render()
            if img is not None:
                # Convert to PIL Image for text overlay
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                
                # Extract ego vehicle info from observation
                obs_reshaped = obs.reshape(15, 6)
                present_vehicles = obs_reshaped[obs_reshaped[:, 0] > 0.5]
                if len(present_vehicles) > 0:
                    ego_vehicle = present_vehicles[0]
                    ego_speed = np.linalg.norm(ego_vehicle[3:5])  # vx, vy
                    ego_speeds.append(ego_speed)
                    
                    # Calculate safety metrics for overlay
                    safety_reward = safety_reward_fn(obs)
                    
                    # Add text overlay
                    draw.text((10, 10), f"Step: {step}", fill=(255, 255, 255))
                    draw.text((10, 30), f"Speed: {ego_speed:.2f} m/s", fill=(255, 255, 255))
                    draw.text((10, 50), f"Safety: {safety_reward:.3f}", fill=(255, 255, 255))
                    draw.text((10, 70), f"Return: {episode_return:.2f}", fill=(255, 255, 255))
                
                images.append(np.asarray(img_pil))
        
        # Sample action from agent
        action, agent = agent.sample_actions(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # Take environment step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Calculate safety metrics
        safety_reward = safety_reward_fn(obs)
        if safety_reward < -0.5:  # Threshold for safety violation
            safety_violations += 1
        
        # Track minimum distance to other vehicles
        obs_reshaped = obs.reshape(15, 6)
        present_vehicles = obs_reshaped[obs_reshaped[:, 0] > 0.5]
        if len(present_vehicles) > 1:
            ego_vehicle = present_vehicles[0]
            other_vehicles = present_vehicles[1:]
            ego_pos = ego_vehicle[1:3]
            other_pos = other_vehicles[:, 1:3]
            distances = np.linalg.norm(other_pos - ego_pos, axis=1)
            min_distances.append(np.min(distances))
        
        # Check for collision
        if reward < -0.9:  # Assuming large negative reward indicates collision
            collision_occurred = True
        
        episode_return += reward
        episode_length += 1
        obs = next_obs
        
        if done or truncated:
            break
    
    # Calculate summary metrics
    avg_min_distance = np.mean(min_distances) if min_distances else 0.0
    avg_ego_speed = np.mean(ego_speeds) if ego_speeds else 0.0
    safety_violation_rate = safety_violations / episode_length if episode_length > 0 else 0.0
    
    trajectory_metrics = {
        "episode_return": episode_return,
        "episode_length": episode_length,
        "collision_occurred": collision_occurred,
        "safety_violations": safety_violations,
        "safety_violation_rate": safety_violation_rate,
        "avg_min_distance": avg_min_distance,
        "avg_ego_speed": avg_ego_speed,
        "min_distance_ever": np.min(min_distances) if min_distances else 0.0,
        "max_ego_speed": np.max(ego_speeds) if ego_speeds else 0.0,
    }
    
    return images, trajectory_metrics


def evaluate_highway_policy(agent, env, num_episodes=10, render_video=False, max_steps=1000):
    """
    Evaluate highway policy over multiple episodes and aggregate results.
    """
    all_metrics = []
    all_videos = []
    
    print(f"Evaluating policy over {num_episodes} episodes...")
    
    for episode in trange(num_episodes, desc="Evaluation"):
        images, metrics = run_highway_trajectory(
            agent, env, max_steps=max_steps, render_video=render_video
        )
        
        metrics["episode_id"] = episode
        all_metrics.append(metrics)
        
        if render_video:
            all_videos.append(images)
    
    # Aggregate statistics
    returns = [m["episode_return"] for m in all_metrics]
    lengths = [m["episode_length"] for m in all_metrics]
    collisions = [m["collision_occurred"] for m in all_metrics]
    safety_rates = [m["safety_violation_rate"] for m in all_metrics]
    min_distances = [m["avg_min_distance"] for m in all_metrics]
    speeds = [m["avg_ego_speed"] for m in all_metrics]
    
    summary_stats = {
        "num_episodes": num_episodes,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "collision_rate": np.mean(collisions),
        "mean_safety_violation_rate": np.mean(safety_rates),
        "mean_min_distance": np.mean(min_distances),
        "mean_ego_speed": np.mean(speeds),
        "success_rate": 1.0 - np.mean(collisions),  # Episodes without collision
    }
    
    return all_metrics, summary_stats, all_videos


def main(_):
    # Highway environment setup
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False
        },
        "action": {
            "type": "ContinuousAction"
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,
        "policy_frequency": 5
    }
    
    # Create environment
    render_mode = "rgb_array" if FLAGS.render else None
    env = gym.make(FLAGS.env_name, config=highway_config, render_mode=render_mode)
    env = FlattenObservation(env)  # Flatten (15, 6) -> (90,)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_steps)
    
    # Create agent
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    kwargs.pop("group_name", None)  # Remove group_name before passing to agent
    kwargs.pop("safety_penalty", None)  # Remove safety_penalty if present
    
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        **kwargs
    )
    
    # Load trained policy
    print(f"Loading policy from: {FLAGS.policy_file}")
    agent = load_highway_agent(agent, FLAGS.policy_file)
    
    # Create output directory
    Path(FLAGS.video_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    episode_metrics, summary_stats, videos = evaluate_highway_policy(
        agent, env, 
        num_episodes=FLAGS.num_episodes,
        render_video=FLAGS.render,
        max_steps=FLAGS.max_steps
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("HIGHWAY POLICY EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {summary_stats['num_episodes']}")
    print(f"Mean Return: {summary_stats['mean_return']:.3f} ± {summary_stats['std_return']:.3f}")
    print(f"Mean Length: {summary_stats['mean_length']:.1f} ± {summary_stats['std_length']:.1f}")
    print(f"Success Rate: {summary_stats['success_rate']:.1%}")
    print(f"Collision Rate: {summary_stats['collision_rate']:.1%}")
    print(f"Mean Safety Violation Rate: {summary_stats['mean_safety_violation_rate']:.1%}")
    print(f"Mean Min Distance: {summary_stats['mean_min_distance']:.2f}m")
    print(f"Mean Ego Speed: {summary_stats['mean_ego_speed']:.2f} m/s")
    
    # Save metrics to JSON
    if FLAGS.save_metrics:
        results = {
            "policy_file": FLAGS.policy_file,
            "evaluation_config": {
                "num_episodes": FLAGS.num_episodes,
                "max_steps": FLAGS.max_steps,
                "env_name": FLAGS.env_name,
                "seed": FLAGS.seed,
            },
            "summary_stats": summary_stats,
            "episode_metrics": episode_metrics,
        }
        
        metrics_file = os.path.join(FLAGS.video_output_dir, "evaluation_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    
    # Save videos
    if FLAGS.render and videos:
        print(f"\nSaving {len(videos)} evaluation videos...")
        policy_name = os.path.basename(FLAGS.policy_file)
        
        for i, images in enumerate(videos):
            if images:  # Check if images were collected
                video_path = os.path.join(
                    FLAGS.video_output_dir,
                    f"highway_eval_{policy_name}_episode_{i}.mp4"
                )
                try:
                    ImageSequenceClip(sequence=images, fps=15).write_videofile(
                        video_path, verbose=False, logger=None
                    )
                    print(f"  Saved: {video_path}")
                except Exception as e:
                    print(f"  Error saving video {i}: {e}")
    
    print(f"\nEvaluation complete! Results saved to: {FLAGS.video_output_dir}")


if __name__ == "__main__":
    app.run(main)