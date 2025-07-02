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
flags.DEFINE_string(
    "video_output_dir", "./evaluation_videos", "Path to the video output directory"
)
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


# Import shared safety functions
from highway_safety_utils import (
    safety_reward_fn,
    is_vehicle_offroad,
    debug_vehicle_position,
    calculate_forward_speed_reward,
    calculate_training_reward,
)


def load_highway_agent(
    agent: Union[SACLearner, DistributionalSACLearner], policy_file: str
):
    """
    Load the highway agent from a checkpoint.
    """
    # Use flexible restoration - include placeholders that will be overwritten if they exist
    # Based on Orbax docs, we need proper placeholder structures
    param_dict = {
        "actor": agent.actor,
        "critic": agent.critic,
        "target_critic_params": agent.target_critic,
        "temp": agent.temp,
        "rng": agent.rng,
        "config": {},  # Use empty dict as placeholder for config restoration
        "training_flags": {},  # Use empty dict as placeholder for flags restoration
    }

    # Add optional components for DistributionalSACLearner
    if hasattr(agent, "limits"):
        param_dict["limits"] = agent.limits
    if hasattr(agent, "q_entropy_lagrange"):
        param_dict["q_entropy_lagrange"] = agent.q_entropy_lagrange

    # Restore from checkpoint
    param_dict = restore_checkpoint(policy_file, target=param_dict)

    # Try to load config from separate pickle file (more reliable)
    checkpoint_config = None
    checkpoint_step = os.path.basename(policy_file).replace("checkpoint_", "")
    config_file = os.path.join(
        os.path.dirname(policy_file), f"config_{checkpoint_step}.pkl"
    )

    if os.path.exists(config_file):
        try:
            import pickle

            with open(config_file, "rb") as f:
                config_data = pickle.load(f)
            checkpoint_config = config_data.get("config", None)
            print("Found config in pickle file")
            print(f"Config keys: {list(checkpoint_config.keys())}")
            if "safety_penalty" in checkpoint_config:
                print(
                    f"safety_penalty from checkpoint: {checkpoint_config['safety_penalty']}"
                )
        except Exception as e:
            print(f"Error loading config pickle: {e}")
            checkpoint_config = None

    # Fallback: check if config is saved in main checkpoint
    if checkpoint_config is None:
        if (
            "config" in param_dict
            and param_dict["config"] is not None
            and len(param_dict["config"]) > 0
        ):
            checkpoint_config = param_dict["config"]
            print("Found config in main checkpoint")
            print(f"Config keys: {list(checkpoint_config.keys())}")
            if "safety_penalty" in checkpoint_config:
                print(
                    f"safety_penalty from checkpoint: {checkpoint_config['safety_penalty']}"
                )
        else:
            print("No config found in checkpoint - using command line flags")

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

    return agent.replace(**replace_dict), checkpoint_config


def run_highway_trajectory(
    agent,
    env: gym.Env,
    max_steps=1000,
    render_video=False,
    safety_bonus_coeff=0.01,
    max_offroad_steps=20,
):
    """
    Run a single highway trajectory with the agent and collect metrics.
    """
    obs, _ = env.reset()
    if hasattr(agent, "env_reset"):
        agent = agent.env_reset(obs)

    images = []
    episode_return = 0
    training_return = 0  # Combined env + safety reward (as used in training)
    episode_length = 0
    safety_violations = 0
    min_distances = []
    ego_speeds = []
    collision_occurred = False
    offroad_violations = 0
    offroad_durations = []
    current_offroad_duration = 0

    # Offroad termination tracking
    # offroad_step_counter = 0
    offroad_terminated = False

    # safety_bonus_coeff passed as parameter from checkpoint config or FLAGS

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
                    safety_reward = safety_reward_fn(obs, env)
                    is_offroad = is_vehicle_offroad(env)
                    debug_info = debug_vehicle_position(env)

                    # Add text overlay with debug info
                    draw.text((10, 10), f"Step: {step}", fill=(255, 255, 255))
                    draw.text(
                        (10, 30), f"Speed: {ego_speed:.2f} m/s", fill=(255, 255, 255)
                    )
                    draw.text(
                        (10, 50), f"Safety: {safety_reward:.3f}", fill=(255, 255, 255)
                    )
                    draw.text(
                        (10, 70),
                        f"Env Return: {episode_return:.2f}",
                        fill=(255, 255, 255),
                    )
                    draw.text(
                        (10, 90),
                        f"Training Return: {training_return:.2f}",
                        fill=(255, 255, 255),
                    )
                    draw.text(
                        (10, 110),
                        f"Offroad: {'YES' if is_offroad else 'NO'}",
                        fill=(255, 0, 0) if is_offroad else (0, 255, 0),
                    )

                    # Add debug information
                    if "lateral" in debug_info and "lane_width" in debug_info:
                        lateral = debug_info["lateral"]
                        lane_width = debug_info["lane_width"]
                        draw.text(
                            (10, 130), f"Lateral: {lateral:.2f}m", fill=(255, 255, 255)
                        )
                        draw.text(
                            (10, 150),
                            f"Lane Width: {lane_width:.2f}m",
                            fill=(255, 255, 255),
                        )
                        draw.text(
                            (10, 170),
                            f"Margin: {lateral - lane_width/2:.2f}m",
                            fill=(
                                (255, 0, 0)
                                if abs(lateral) > lane_width / 2
                                else (0, 255, 0)
                            ),
                        )

                    if "on_road_property" in debug_info:
                        draw.text(
                            (10, 190),
                            f"OnRoad Prop: {debug_info['on_road_property']}",
                            fill=(255, 255, 255),
                        )

                images.append(np.asarray(img_pil))

        # Sample action from agent
        action, agent = agent.sample_actions(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Take environment step
        next_obs, reward, done, truncated, info = env.step(action)

        # Use shared training reward calculation
        training_reward, reward_components = calculate_training_reward(
            env, reward, info, safety_bonus_coeff, next_obs=next_obs
        )
        safety_reward = reward_components["safety_reward"]
        
        # Calculate safety metrics
        if safety_reward < -0.5:  # Threshold for safety violation
            safety_violations += 1

        # Track offroad status
        # is_offroad = is_vehicle_offroad(env)
        # if is_offroad:
        #     offroad_violations += 1
        #     current_offroad_duration += 1
        #     # Check for offroad termination
        #     offroad_step_counter += 1
        #     if offroad_step_counter >= max_offroad_steps:
        #         done = True
        #         offroad_terminated = True
        # else:
        #     if current_offroad_duration > 0:
        #         offroad_durations.append(current_offroad_duration)
        #         current_offroad_duration = 0
        #     # Reset offroad step counter when back on road
        #     offroad_step_counter = 0

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
        if info["rewards"]["collision_reward"] < 0.0:
            collision_occurred = True

        episode_return += reward
        training_return += training_reward
        episode_length += 1
        obs = next_obs

        if done or truncated:
            is_offroad = is_vehicle_offroad(env)
            if is_offroad:
                offroad_terminated = True
            break

    # Handle final offroad duration if episode ended while offroad
    if current_offroad_duration > 0:
        offroad_durations.append(current_offroad_duration)

    # Calculate summary metrics
    avg_min_distance = float(np.mean(min_distances)) if min_distances else 0.0
    avg_ego_speed = float(np.mean(ego_speeds)) if ego_speeds else 0.0
    safety_violation_rate = (
        float(safety_violations / episode_length) if episode_length > 0 else 0.0
    )
    offroad_violation_rate = (
        float(offroad_violations / episode_length) if episode_length > 0 else 0.0
    )
    avg_offroad_duration = (
        float(np.mean(offroad_durations)) if offroad_durations else 0.0
    )

    trajectory_metrics = {
        "episode_return": float(episode_return),
        "training_return": float(training_return),
        "episode_length": int(episode_length),
        "collision_occurred": bool(collision_occurred),
        "safety_violations": int(safety_violations),
        "safety_violation_rate": safety_violation_rate,
        "avg_min_distance": avg_min_distance,
        "avg_ego_speed": avg_ego_speed,
        "min_distance_ever": float(np.min(min_distances)) if min_distances else 0.0,
        "max_ego_speed": float(np.max(ego_speeds)) if ego_speeds else 0.0,
        "offroad_violations": int(offroad_violations),
        "offroad_violation_rate": offroad_violation_rate,
        "avg_offroad_duration": avg_offroad_duration,
        "total_offroad_steps": int(offroad_violations),
        "offroad_terminated": bool(offroad_terminated),
    }

    return images, trajectory_metrics


def evaluate_highway_policy(
    agent,
    env,
    num_episodes=10,
    render_video=False,
    max_steps=1000,
    safety_bonus_coeff=0.01,
    max_offroad_steps=20,
):
    """
    Evaluate highway policy over multiple episodes and aggregate results.
    """
    all_metrics = []
    all_videos = []

    print(f"Evaluating policy over {num_episodes} episodes...")

    for episode in trange(num_episodes, desc="Evaluation"):
        images, metrics = run_highway_trajectory(
            agent,
            env,
            max_steps=max_steps,
            render_video=render_video,
            safety_bonus_coeff=safety_bonus_coeff,
            max_offroad_steps=max_offroad_steps,
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
    offroad_rates = [m["offroad_violation_rate"] for m in all_metrics]
    offroad_durations = [m["avg_offroad_duration"] for m in all_metrics]
    total_offroad_steps = [m["total_offroad_steps"] for m in all_metrics]
    offroad_terminations = [m["offroad_terminated"] for m in all_metrics]

    summary_stats = {
        "num_episodes": int(num_episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "collision_rate": float(np.mean(collisions)),
        "mean_safety_violation_rate": float(np.mean(safety_rates)),
        "mean_min_distance": float(np.mean(min_distances)),
        "mean_ego_speed": float(np.mean(speeds)),
        "success_rate": float(1.0 - np.mean(collisions)),  # Episodes without collision
        "mean_offroad_violation_rate": float(np.mean(offroad_rates)),
        "mean_offroad_duration": float(np.mean(offroad_durations)),
        "total_offroad_steps_all_episodes": int(np.sum(total_offroad_steps)),
        "offroad_episode_rate": float(
            np.mean([1.0 if rate > 0 else 0.0 for rate in offroad_rates])
        ),
        "offroad_termination_rate": float(np.mean(offroad_terminations)),
    }

    return all_metrics, summary_stats, all_videos


def main(_):
    # Highway environment setup
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [30, 45],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": False,  # Keep False to avoid early termination, use our enhanced detection
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
    kwargs.pop(
        "max_offroad_steps", None
    )  # Remove max_offroad_steps before passing to agent

    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    # Load trained policy
    policy_path = os.path.abspath(FLAGS.policy_file)
    print(f"Loading policy from: {policy_path}")
    agent, checkpoint_config = load_highway_agent(agent, policy_path)

    # Use checkpoint config if available, otherwise fall back to FLAGS.config
    if checkpoint_config is not None:
        print("Using config from checkpoint")
        # Extract safety_penalty from checkpoint config for evaluation
        safety_bonus_coeff = checkpoint_config.get("safety_penalty", 0.01)
        max_offroad_steps = checkpoint_config.get("max_offroad_steps", 20)
    else:
        print("Using config from command line flags")
        safety_bonus_coeff = FLAGS.config.get("safety_penalty", 0.01)
        max_offroad_steps = FLAGS.config.get("max_offroad_steps", 20)

    # Create output directory
    Path(FLAGS.video_output_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluation
    episode_metrics, summary_stats, videos = evaluate_highway_policy(
        agent,
        env,
        num_episodes=FLAGS.num_episodes,
        render_video=FLAGS.render,
        max_steps=FLAGS.max_steps,
        safety_bonus_coeff=safety_bonus_coeff,
        max_offroad_steps=max_offroad_steps,
    )

    # Print summary statistics
    print("\n" + "=" * 50)
    print("HIGHWAY POLICY EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {summary_stats['num_episodes']}")
    print(
        f"Mean Return: {summary_stats['mean_return']:.3f} ± {summary_stats['std_return']:.3f}"
    )
    print(
        f"Mean Length: {summary_stats['mean_length']:.1f} ± {summary_stats['std_length']:.1f}"
    )
    print(f"Success Rate: {summary_stats['success_rate']:.1%}")
    print(f"Collision Rate: {summary_stats['collision_rate']:.1%}")
    print(
        f"Mean Safety Violation Rate: {summary_stats['mean_safety_violation_rate']:.1%}"
    )
    print(f"Mean Min Distance: {summary_stats['mean_min_distance']:.2f}m")
    print(f"Mean Ego Speed: {summary_stats['mean_ego_speed']:.2f} m/s")
    print("=" * 50)
    print("OFFROAD SAFETY METRICS")
    print("=" * 50)
    print(
        f"Mean Offroad Violation Rate: {summary_stats['mean_offroad_violation_rate']:.1%}"
    )
    print(f"Offroad Episode Rate: {summary_stats['offroad_episode_rate']:.1%}")
    print(f"Mean Offroad Duration: {summary_stats['mean_offroad_duration']:.1f} steps")
    print(f"Total Offroad Steps: {summary_stats['total_offroad_steps_all_episodes']}")
    print(f"Offroad Termination Rate: {summary_stats['offroad_termination_rate']:.1%}")

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
                    f"highway_eval_{policy_name}_episode_{i}.mp4",
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
