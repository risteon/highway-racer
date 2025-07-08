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
import jax.numpy as jnp
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
flags.DEFINE_string("output_dir", "./q_value_analysis", "Path to the output directory")
flags.DEFINE_integer("num_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_integer("max_steps", 1000, "Maximum steps per episode")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_boolean("render", True, "Enable video rendering and frame saving")
flags.DEFINE_string("env_name", "highway-v0", "Highway environment name")
flags.DEFINE_integer("num_vehicles", 10, "Number of vehicles in environment")

config_flags.DEFINE_config_file(
    "config",
    "configs/highway_distributional_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
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
                if "cvar_risk" in checkpoint_config:
                    print(
                        f"cvar_risk from checkpoint: {checkpoint_config['cvar_risk']}"
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


def get_q_distributions(agent, obs, actions):
    """
    Get Q-value distributions for given observations and actions.

    Args:
        agent: Trained agent with critic network
        obs: Observation (flattened)
        actions: Actions to evaluate (batch of actions)

    Returns:
        q_logits: Raw Q-distribution logits
        q_atoms: Q-value atoms (support points)
        q_probs: Q-distribution probabilities (softmax of logits)
    """
    # Ensure obs and actions have correct batch dimensions
    if obs.ndim == 1:
        obs = obs[None, :]  # Add batch dimension
    if actions.ndim == 1:
        actions = actions[None, :]  # Add batch dimension

    # Get Q-value distributions from critic
    q_logits, q_atoms = agent.critic.apply_fn(
        {"params": agent.critic.params}, obs, actions
    )

    # Convert logits to probabilities
    q_probs = jax.nn.softmax(q_logits, axis=-1)

    return q_logits, q_atoms, q_probs


def run_episode_with_q_recording(agent, env, max_steps=1000, render_frames=True):
    """
    Run a single episode while recording Q-value distributions and frames.

    Returns:
        episode_data: Dict containing all recorded data
    """
    obs, info = env.reset()

    # Storage for episode data
    episode_data = {
        "observations": [],
        "actions_taken": [],
        "rewards": [],
        # Speed control actions
        "q_logits_accelerate": [],
        "q_atoms_accelerate": [],
        "q_probs_accelerate": [],
        "q_logits_brake": [],
        "q_atoms_brake": [],
        "q_probs_brake": [],
        "q_logits_continue": [],  # No acceleration/braking
        "q_atoms_continue": [],
        "q_probs_continue": [],
        # Steering actions
        "q_logits_steer_right": [],
        "q_atoms_steer_right": [],
        "q_probs_steer_right": [],
        "q_logits_steer_left": [],
        "q_atoms_steer_left": [],
        "q_probs_steer_left": [],
        # Policy action (what agent actually chose)
        "q_logits_policy": [],
        "q_atoms_policy": [],
        "q_probs_policy": [],
        "frames": [],
        "step_rewards": [],
    }

    episode_return = 0.0
    episode_length = 0

    for step in range(max_steps):
        # Store current observation
        episode_data["observations"].append(obs.copy())

        # Render frame if requested
        if render_frames:
            frame = env.render()
            if frame is not None:
                episode_data["frames"].append(frame)

        # Sample action from policy
        if hasattr(agent, "sample_actions"):
            # For newer agent interface
            action, agent = agent.sample_actions(obs)
            action = np.array(action)
        else:
            # For older agent interface - use actor directly
            dist = agent.actor.apply_fn({"params": agent.actor.params}, obs[None, :])
            action = dist.sample(seed=jax.random.PRNGKey(step))
            action = np.array(action[0])  # Remove batch dimension

        episode_data["actions_taken"].append(action.copy())

        # Define test actions for comprehensive highway driving analysis
        # Highway-env action space: [steering, acceleration]
        test_actions = {
            # Speed control actions
            "accelerate": jnp.array([[0.0, 1.0]]),  # No steering, full acceleration
            "brake": jnp.array([[0.0, -1.0]]),  # No steering, full braking
            "continue": jnp.array([[0.0, 0.0]]),  # No steering, no acceleration
            # Steering actions
            "steer_right": jnp.array(
                [[1.0, 0.0]]
            ),  # Full right steering, no acceleration
            "steer_left": jnp.array(
                [[-1.0, 0.0]]
            ),  # Full left steering, no acceleration
            # Policy action
            "policy": action[None, :],  # Action actually taken by agent
        }

        obs_batch = obs[None, :]  # Add batch dimension for critic evaluation

        # Get Q-distributions for all test actions
        for action_name, action_array in test_actions.items():
            q_logits, q_atoms, q_probs = get_q_distributions(
                agent, obs_batch, action_array
            )
            episode_data[f"q_logits_{action_name}"].append(q_logits)
            episode_data[f"q_atoms_{action_name}"].append(q_atoms)
            episode_data[f"q_probs_{action_name}"].append(q_probs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_data["step_rewards"].append(reward)
        episode_return += reward
        episode_length += 1

        obs = next_obs

        if terminated or truncated:
            break

    # Convert lists to numpy arrays for easier analysis
    for key in ["observations", "actions_taken", "step_rewards"]:
        if episode_data[key]:
            episode_data[key] = np.array(episode_data[key])

    # Convert JAX arrays to numpy for Q-value data
    q_value_keys = [
        # Speed control actions
        "q_logits_accelerate",
        "q_atoms_accelerate",
        "q_probs_accelerate",
        "q_logits_brake",
        "q_atoms_brake",
        "q_probs_brake",
        "q_logits_continue",
        "q_atoms_continue",
        "q_probs_continue",
        # Steering actions
        "q_logits_steer_right",
        "q_atoms_steer_right",
        "q_probs_steer_right",
        "q_logits_steer_left",
        "q_atoms_steer_left",
        "q_probs_steer_left",
        # Policy action
        "q_logits_policy",
        "q_atoms_policy",
        "q_probs_policy",
    ]

    for key in q_value_keys:
        if episode_data[key]:
            episode_data[key] = np.array([np.array(x) for x in episode_data[key]])

    episode_data["episode_return"] = episode_return
    episode_data["episode_length"] = episode_length

    return episode_data


def save_episode_data(episode_data, output_dir, episode_idx, checkpoint_config=None):
    """
    Save episode data to NPZ file and optionally create video.
    """
    # Create episode-specific output directory
    episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
    Path(episode_dir).mkdir(parents=True, exist_ok=True)

    # Save Q-value data and frames to NPZ file
    npz_data = {
        "observations": episode_data["observations"],
        "actions_taken": episode_data["actions_taken"],
        "step_rewards": episode_data["step_rewards"],
        # Speed control actions
        "q_logits_accelerate": episode_data["q_logits_accelerate"],
        "q_atoms_accelerate": episode_data["q_atoms_accelerate"],
        "q_probs_accelerate": episode_data["q_probs_accelerate"],
        "q_logits_brake": episode_data["q_logits_brake"],
        "q_atoms_brake": episode_data["q_atoms_brake"],
        "q_probs_brake": episode_data["q_probs_brake"],
        "q_logits_continue": episode_data["q_logits_continue"],
        "q_atoms_continue": episode_data["q_atoms_continue"],
        "q_probs_continue": episode_data["q_probs_continue"],
        # Steering actions
        "q_logits_steer_right": episode_data["q_logits_steer_right"],
        "q_atoms_steer_right": episode_data["q_atoms_steer_right"],
        "q_probs_steer_right": episode_data["q_probs_steer_right"],
        "q_logits_steer_left": episode_data["q_logits_steer_left"],
        "q_atoms_steer_left": episode_data["q_atoms_steer_left"],
        "q_probs_steer_left": episode_data["q_probs_steer_left"],
        # Policy action
        "q_logits_policy": episode_data["q_logits_policy"],
        "q_atoms_policy": episode_data["q_atoms_policy"],
        "q_probs_policy": episode_data["q_probs_policy"],
        # Environment data
        "frames": (
            np.array(episode_data["frames"]) if episode_data["frames"] else np.array([])
        ),
        "episode_return": episode_data["episode_return"],
        "episode_length": episode_data["episode_length"],
    }

    npz_file = os.path.join(episode_dir, "q_values_and_trajectory.npz")
    np.savez_compressed(npz_file, **npz_data)
    print(f"  Saved Q-value data: {npz_file}")

    # Save video if frames were recorded
    if episode_data["frames"] and len(episode_data["frames"]) > 0:
        video_file = os.path.join(episode_dir, f"episode_{episode_idx}_video.mp4")
        try:
            ImageSequenceClip(sequence=episode_data["frames"], fps=15).write_videofile(
                video_file, verbose=False, logger=None
            )
            print(f"  Saved video: {video_file}")
        except Exception as e:
            print(f"  Error saving video: {e}")

    # Save summary info as JSON
    summary = {
        "episode_idx": episode_idx,
        "episode_return": float(episode_data["episode_return"]),
        "episode_length": int(episode_data["episode_length"]),
        "num_frames": len(episode_data["frames"]),
        "q_value_shape": {
            "accelerate": (
                episode_data["q_probs_accelerate"][0].shape
                if len(episode_data["q_probs_accelerate"]) > 0
                else None
            ),
            "brake": (
                episode_data["q_probs_brake"][0].shape
                if len(episode_data["q_probs_brake"]) > 0
                else None
            ),
            "continue": (
                episode_data["q_probs_continue"][0].shape
                if len(episode_data["q_probs_continue"]) > 0
                else None
            ),
            "steer_right": (
                episode_data["q_probs_steer_right"][0].shape
                if len(episode_data["q_probs_steer_right"]) > 0
                else None
            ),
            "steer_left": (
                episode_data["q_probs_steer_left"][0].shape
                if len(episode_data["q_probs_steer_left"]) > 0
                else None
            ),
            "policy": (
                episode_data["q_probs_policy"][0].shape
                if len(episode_data["q_probs_policy"]) > 0
                else None
            ),
        },
        "checkpoint_cvar_risk": (
            checkpoint_config.get("cvar_risk") if checkpoint_config else None
        ),
    }

    summary_file = os.path.join(episode_dir, "episode_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


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
            "vehicles_count": 8,
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
        "offroad_terminal": True,
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

    # Check if output directory already contains run/checkpoint structure
    # If so, use it directly; otherwise auto-generate subdirectories
    if os.path.basename(FLAGS.output_dir) in ["data", "plots"]:
        # bash script passes structured path like ./q_values/run/ckpt/data
        auto_output_dir = FLAGS.output_dir
        print(f"Using structured output directory: {auto_output_dir}")
    else:
        # Standalone usage - auto-generate output folder based on run name and checkpoint name
        checkpoint_name = os.path.basename(FLAGS.policy_file)
        if not checkpoint_name:
            checkpoint_name = "unknown_checkpoint"

        # Extract run name from policy path (parent directory of checkpoint)
        run_name = os.path.basename(os.path.dirname(FLAGS.policy_file))
        if not run_name:
            run_name = "unknown_run"

        # Create output folder under specified directory
        auto_output_dir = os.path.join(FLAGS.output_dir, run_name, checkpoint_name)
        print(f"Auto-generated output directory: {auto_output_dir}")

    # Default highway environment configuration
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        "vehicles_count": FLAGS.num_vehicles,
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -10,
        "reward_speed_range": [10, 40],  # Default, may be overridden by checkpoint
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": True,  # Keep False to avoid early termination
    }

    # Override environment config with checkpoint configs if available
    if checkpoint_highway_config is not None:
        print("Using highway environment config from checkpoint")
        # Update the highway config with values from checkpoint
        for key, value in checkpoint_highway_config.items():
            highway_config[key] = value
            print(f"Using {key} from checkpoint highway config: {value}")

    # Create environment with potentially updated config
    render_mode = "rgb_array" if FLAGS.render else None
    env = gym.make(FLAGS.env_name, config=highway_config, render_mode=render_mode)
    env = FlattenObservation(env)  # Flatten (15, 6) -> (90,)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    # Create output directory
    Path(auto_output_dir).mkdir(parents=True, exist_ok=True)

    # Check if agent has distributional critic
    has_distributional_critic = hasattr(agent, "critic") and hasattr(
        agent.critic, "apply_fn"
    )
    if not has_distributional_critic:
        print("Warning: Agent does not appear to have a distributional critic!")
        print("This script is designed for DistributionalSACLearner agents.")
        return

    print(f"\nRecording Q-value distributions over {FLAGS.num_episodes} episodes...")
    print("Actions analyzed:")
    print(
        "  Speed Control: Accelerate [0.0, 1.0], Brake [0.0, -1.0], Continue [0.0, 0.0]"
    )
    print("  Steering: Steer Right [1.0, 0.0], Steer Left [-1.0, 0.0]")
    print("  Policy: Action actually chosen by agent")

    # Run episodes and record Q-value data
    all_episodes_data = []

    for episode in trange(FLAGS.num_episodes, desc="Recording episodes"):
        print(f"\nEpisode {episode + 1}/{FLAGS.num_episodes}")

        # Run episode with Q-value recording
        episode_data = run_episode_with_q_recording(
            agent, env, max_steps=FLAGS.max_steps, render_frames=FLAGS.render
        )

        all_episodes_data.append(episode_data)

        # Save episode data
        save_episode_data(episode_data, auto_output_dir, episode, checkpoint_config)

        print(f"  Episode return: {episode_data['episode_return']:.2f}")
        print(f"  Episode length: {episode_data['episode_length']}")

    # Save aggregate analysis data
    aggregate_data = {
        "policy_file": FLAGS.policy_file,
        "config": {
            "num_episodes": FLAGS.num_episodes,
            "max_steps": FLAGS.max_steps,
            "env_name": FLAGS.env_name,
            "num_vehicles": FLAGS.num_vehicles,
            "seed": FLAGS.seed,
        },
        "episodes_summary": [
            {
                "episode_idx": i,
                "episode_return": float(ep["episode_return"]),
                "episode_length": int(ep["episode_length"]),
            }
            for i, ep in enumerate(all_episodes_data)
        ],
        "mean_return": float(
            np.mean([ep["episode_return"] for ep in all_episodes_data])
        ),
        "std_return": float(np.std([ep["episode_return"] for ep in all_episodes_data])),
        "mean_length": float(
            np.mean([ep["episode_length"] for ep in all_episodes_data])
        ),
    }

    aggregate_file = os.path.join(auto_output_dir, "aggregate_analysis.json")
    with open(aggregate_file, "w") as f:
        json.dump(aggregate_data, f, indent=2)

    print(f"\n" + "=" * 60)
    print("Q-VALUE RECORDING COMPLETE")
    print("=" * 60)
    print(f"Episodes recorded: {FLAGS.num_episodes}")
    print(
        f"Mean return: {aggregate_data['mean_return']:.3f} ± {aggregate_data['std_return']:.3f}"
    )
    print(f"Mean length: {aggregate_data['mean_length']:.1f}")
    print(f"Output directory: {auto_output_dir}")
    print("\nFiles saved per episode:")
    print("  - q_values_and_trajectory.npz (Q-distributions and trajectory data)")
    print("  - episode_X_video.mp4 (environment video)")
    print("  - episode_summary.json (episode metadata)")
    print(f"\nAggregate analysis: {aggregate_file}")
    print("\nUse the NPZ files to analyze Q-value distributions over time.")


if __name__ == "__main__":
    app.run(main)
