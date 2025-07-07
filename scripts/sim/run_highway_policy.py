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
flags.DEFINE_integer("num_vehicles", 50, "Number of vehicles in environment")

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

# Import shared trajectory utilities
from highway_trajectory_utils import (
    run_highway_trajectory,
    LearnedPolicyAgent,
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
            highway_env_config = config_data.get("highway_env_config", None)
            print("Found config in pickle file")
            if checkpoint_config:
                print(f"Config keys: {list(checkpoint_config.keys())}")
                if "safety_penalty" in checkpoint_config:
                    print(
                        f"safety_penalty from checkpoint: {checkpoint_config['safety_penalty']}"
                    )
            if highway_env_config:
                print("Found highway environment config in checkpoint")
        except Exception as e:
            print(f"Error loading config pickle: {e}")
            checkpoint_config = None
            highway_env_config = None

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

    return agent.replace(**replace_dict), checkpoint_config, highway_env_config


def run_highway_trajectory_wrapper(
    agent,
    env: gym.Env,
    highway_config,
    max_steps=1000,
    render_video=False,
    safety_bonus_coeff=0.01,
    video_output_dir="./evaluation_videos",
):
    """
    Wrapper for the shared run_highway_trajectory function to maintain backward compatibility.

    This function wraps the learned policy agent and calls the shared trajectory analysis.
    """
    # Wrap the learned policy in the AgentInterface
    agent_wrapper = LearnedPolicyAgent(agent, eval_mode=True)

    # Set video output path if rendering
    video_path = None
    if render_video:
        video_path = f"{video_output_dir}/highway_trajectory_episode.mp4"

    # Use the shared trajectory analysis function
    trajectory_metrics, images = run_highway_trajectory(
        agent=agent_wrapper,
        env=env,
        highway_config=highway_config,
        max_steps=max_steps,
        render_video=render_video,
        safety_bonus_coeff=safety_bonus_coeff,
        video_output_path=video_path,
        analysis_mode="evaluation",
    )

    return images, trajectory_metrics


def evaluate_highway_policy(
    agent,
    env,
    highway_config,
    num_episodes=10,
    render_video=False,
    max_steps=1000,
    safety_bonus_coeff=0.01,
    video_output_dir="./evaluation_videos",
):
    """
    Evaluate highway policy over multiple episodes and aggregate results.
    """
    all_metrics = []
    all_videos = []

    print(f"Evaluating policy over {num_episodes} episodes...")

    for episode in trange(num_episodes, desc="Evaluation"):
        images, metrics = run_highway_trajectory_wrapper(
            agent,
            env,
            highway_config,
            max_steps=max_steps,
            render_video=render_video,
            safety_bonus_coeff=safety_bonus_coeff,
            video_output_dir=video_output_dir,
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
    }

    return all_metrics, summary_stats, all_videos


def main(_):
    # Create agent first to load checkpoint config
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    kwargs.pop("group_name", None)  # Remove group_name before passing to agent
    kwargs.pop("safety_penalty", None)  # Remove safety_penalty if present
    kwargs.pop(
        "max_offroad_steps", None
    )  # Remove max_offroad_steps before passing to agent

    # Default highway environment configuration for initial env creation
    temp_highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        "vehicles_count": FLAGS.num_vehicles,
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [30, 45],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": False,
    }

    # Temporarily create environment to get observation/action spaces for agent creation
    temp_env = gym.make(FLAGS.env_name, config=temp_highway_config, render_mode=None)
    temp_env = FlattenObservation(temp_env)

    agent = globals()[model_cls].create(
        FLAGS.seed, temp_env.observation_space, temp_env.action_space, **kwargs
    )
    temp_env.close()

    # Load trained policy and get checkpoint config
    policy_path = os.path.abspath(FLAGS.policy_file)
    print(f"Loading policy from: {policy_path}")
    agent, checkpoint_config, checkpoint_highway_config = load_highway_agent(
        agent, policy_path
    )

    # Auto-generate output folder based on run name and checkpoint name
    checkpoint_name = os.path.basename(FLAGS.policy_file)
    if not checkpoint_name:
        checkpoint_name = "unknown_checkpoint"

    # Extract run name from policy path (parent directory of checkpoint)
    run_name = os.path.basename(os.path.dirname(FLAGS.policy_file))
    if not run_name:
        run_name = "unknown_run"

    # Create evaluation folder under ./evaluation/<run-name>/<checkpoint-name>/
    auto_output_dir = os.path.join("evaluation", run_name, checkpoint_name)

    # Use auto-generated path instead of command line flag
    actual_output_dir = auto_output_dir
    print(f"Auto-generated output directory: {actual_output_dir}")

    # Default highway environment configuration
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        "vehicles_count": FLAGS.num_vehicles,
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [30, 45],  # Default, may be overridden by checkpoint
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": False,  # Keep False to avoid early termination, use our enhanced detection
    }

    # Override environment config with checkpoint configs if available
    if checkpoint_config is not None:
        print("Using config from checkpoint")
        # Extract training config for evaluation
        safety_bonus_coeff = checkpoint_config.get("safety_penalty", 0.01)
        max_offroad_steps = checkpoint_config.get("max_offroad_steps", 20)
    else:
        print("Using config from command line flags")
        safety_bonus_coeff = FLAGS.config.get("safety_penalty", 0.01)
        max_offroad_steps = FLAGS.config.get("max_offroad_steps", 20)

    # Use checkpoint highway environment config if available (priority over algorithm config)
    if checkpoint_highway_config is not None:
        print("Using highway environment config from checkpoint")
        # Update the highway config with values from checkpoint
        for key, value in checkpoint_highway_config.items():
            highway_config[key] = value
            print(f"Using {key} from checkpoint highway config: {value}")
    else:
        print("No highway environment config in checkpoint, using defaults")

        # Fallback: check algorithm config for environment settings (legacy)
        if checkpoint_config is not None:
            env_overrides = {
                "reward_speed_range": checkpoint_config.get("reward_speed_range"),
                "collision_reward": checkpoint_config.get("collision_reward"),
                "right_lane_reward": checkpoint_config.get("right_lane_reward"),
                "high_speed_reward": checkpoint_config.get("high_speed_reward"),
                "lane_change_reward": checkpoint_config.get("lane_change_reward"),
                "normalize_reward": checkpoint_config.get("normalize_reward"),
            }

            for key, value in env_overrides.items():
                if value is not None:
                    highway_config[key] = value
                    print(f"Using {key} from checkpoint algorithm config: {value}")

    # TEST IT OUT
    highway_config["vehicles_count"] = FLAGS.num_vehicles
    # highway_config["vehicles_density"] = 0.5

    # Create environment with potentially updated config
    render_mode = "rgb_array" if FLAGS.render else None
    env = gym.make(FLAGS.env_name, config=highway_config, render_mode=render_mode)
    env = FlattenObservation(env)  # Flatten (15, 6) -> (90,)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    # Create output directory
    Path(actual_output_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluation
    episode_metrics, summary_stats, videos = evaluate_highway_policy(
        agent,
        env,
        highway_config,
        num_episodes=FLAGS.num_episodes,
        render_video=FLAGS.render,
        max_steps=FLAGS.max_steps,
        safety_bonus_coeff=safety_bonus_coeff,
        video_output_dir=actual_output_dir,
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

        metrics_file = os.path.join(actual_output_dir, "evaluation_metrics.json")
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
                    actual_output_dir,
                    f"highway_eval_{policy_name}_episode_{i}.mp4",
                )
                try:
                    ImageSequenceClip(sequence=images, fps=15).write_videofile(
                        video_path, verbose=False, logger=None
                    )
                    print(f"  Saved: {video_path}")
                except Exception as e:
                    print(f"  Error saving video {i}: {e}")

    print(f"\nEvaluation complete! Results saved to: {actual_output_dir}")


if __name__ == "__main__":
    app.run(main)
