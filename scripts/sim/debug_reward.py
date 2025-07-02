#!/usr/bin/env python
"""
Debug script to analyze reward calculation for a single trajectory.
"""
import warnings
import os
import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import TimeLimit, FlattenObservation
import jax
from ml_collections import config_flags
from flax.training.checkpoints import restore_checkpoint
from absl import app, flags

from jaxrl5.agents import DistributionalSACLearner

# Import shared safety functions
from highway_safety_utils import (
    safety_reward_fn,
    is_vehicle_offroad,
    debug_vehicle_position,
    calculate_forward_speed_reward,
    calculate_training_reward,
)

jax.config.update("jax_platform_name", "cpu")
warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_file", None, "Path to the policy checkpoint directory")
flags.DEFINE_integer("max_steps", 50, "Maximum steps for debug")
flags.DEFINE_integer("seed", 42, "Random seed")

config_flags.DEFINE_config_file(
    "config",
    "scripts/sim/configs/highway_distributional_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def load_highway_agent(agent, policy_file: str):
    """Load the highway agent from a checkpoint."""
    param_dict = {
        "actor": agent.actor,
        "critic": agent.critic,
        "target_critic_params": agent.target_critic,
        "temp": agent.temp,
        "rng": agent.rng,
        "config": {},
        "training_flags": {},
    }

    if hasattr(agent, "limits"):
        param_dict["limits"] = agent.limits
    if hasattr(agent, "q_entropy_lagrange"):
        param_dict["q_entropy_lagrange"] = agent.q_entropy_lagrange

    param_dict = restore_checkpoint(policy_file, target=param_dict)

    # Try to load config from pickle file
    checkpoint_config = None
    highway_env_config = None
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
            if highway_env_config:
                print("Found highway environment config in checkpoint")
        except Exception as e:
            print(f"Error loading config pickle: {e}")

    replace_dict = {
        "actor": param_dict["actor"],
        "critic": param_dict["critic"],
        "target_critic": param_dict["target_critic_params"],
        "temp": param_dict["temp"],
        "rng": param_dict["rng"],
    }

    if "limits" in param_dict:
        replace_dict["limits"] = param_dict["limits"]
    if "q_entropy_lagrange" in param_dict:
        replace_dict["q_entropy_lagrange"] = param_dict["q_entropy_lagrange"]

    return agent.replace(**replace_dict), checkpoint_config, highway_env_config


def debug_single_trajectory(agent, env, max_steps=50, safety_bonus_coeff=0.01):
    """Run a single trajectory with detailed reward debugging."""
    print("=" * 80)
    print("STARTING REWARD DEBUG TRAJECTORY")
    print("=" * 80)
    
    obs, _ = env.reset()
    if hasattr(agent, "env_reset"):
        agent = agent.env_reset(obs)

    episode_return = 0
    training_return = 0
    cumulative_env_return = 0
    cumulative_training_return = 0

    print(f"Initial observation shape: {obs.shape}")
    print(f"Safety bonus coefficient: {safety_bonus_coeff}")
    print()

    for step in range(max_steps):
        print(f"--- STEP {step} ---")
        
        # Extract ego vehicle info from observation
        obs_reshaped = obs.reshape(15, 6)
        present_vehicles = obs_reshaped[obs_reshaped[:, 0] > 0.5]
        
        if len(present_vehicles) > 0:
            ego_vehicle = present_vehicles[0]
            ego_pos = ego_vehicle[1:3]  # [x, y]
            ego_vel = ego_vehicle[3:5]  # [vx, vy]
            ego_speed = np.linalg.norm(ego_vel)
            ego_heading = ego_vehicle[5]
            
            print(f"Ego position: [{ego_pos[0]:.2f}, {ego_pos[1]:.2f}]")
            print(f"Ego velocity: [{ego_vel[0]:.2f}, {ego_vel[1]:.2f}] (speed: {ego_speed:.2f})")
            print(f"Ego heading: {ego_heading:.3f} rad")
            
            # Check if vehicle is moving backward (negative x velocity)
            is_backward = ego_vel[0] < 0
            print(f"Moving backward: {is_backward}")

        # Sample action from agent
        action, agent = agent.sample_actions(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        print(f"Action: [{action[0]:.3f}, {action[1]:.3f}] (steering, acceleration)")

        # Take environment step
        next_obs, env_reward, done, truncated, info = env.step(action)
        
        print(f"Environment reward: {env_reward:.4f}")
        
        # Print detailed reward breakdown from info
        if "rewards" in info:
            reward_dict = info["rewards"]
            print("Highway-env reward components:")
            for key, value in reward_dict.items():
                print(f"  {key}: {value:.4f}")
        
        # Calculate our training reward using the shared function
        training_reward, reward_components = calculate_training_reward(
            env, env_reward, info, safety_bonus_coeff, next_obs=next_obs
        )
        
        print(f"Training reward: {training_reward:.4f}")
        print("Training reward components:")
        for key, value in reward_components.items():
            print(f"  {key}: {value:.4f}")
        
        # Debug vehicle position
        debug_info = debug_vehicle_position(env)
        print("Vehicle position debug:")
        for key, value in debug_info.items():
            if key not in ["position"]:  # Skip position as we already printed it
                print(f"  {key}: {value}")
        
        # Update cumulative returns
        cumulative_env_return += env_reward
        cumulative_training_return += training_reward
        
        print(f"Cumulative env return: {cumulative_env_return:.4f}")
        print(f"Cumulative training return: {cumulative_training_return:.4f}")
        
        # Check for concerning patterns
        if env_reward > 0 and is_backward:
            print("⚠️  WARNING: Positive environment reward while moving backward!")
        
        if training_reward > env_reward + 0.1:  # Significant difference
            print("⚠️  WARNING: Training reward much higher than env reward!")
            
        print()
        
        obs = next_obs
        
        if done or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print("=" * 80)
    print("TRAJECTORY SUMMARY")
    print("=" * 80)
    print(f"Total steps: {step + 1}")
    print(f"Final cumulative env return: {cumulative_env_return:.4f}")
    print(f"Final cumulative training return: {cumulative_training_return:.4f}")
    print(f"Difference: {cumulative_training_return - cumulative_env_return:.4f}")


def main(_):
    # Create agent first to load checkpoint config
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    kwargs.pop("group_name", None)
    kwargs.pop("safety_penalty", None)
    kwargs.pop("max_offroad_steps", None)  # Remove highway-specific parameter

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
        "vehicles_count": 50,
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [30, 45],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": False,
    }
    
    # Temporarily create environment to get observation/action spaces for agent creation
    temp_env = gym.make("highway-v0", config=temp_highway_config, render_mode=None)
    temp_env = FlattenObservation(temp_env)
    
    agent = globals()[model_cls].create(
        FLAGS.seed, temp_env.observation_space, temp_env.action_space, **kwargs
    )
    temp_env.close()

    # Load trained policy and get checkpoint config
    policy_path = os.path.abspath(FLAGS.policy_file)
    print(f"Loading policy from: {policy_path}")
    agent, checkpoint_config, checkpoint_highway_config = load_highway_agent(agent, policy_path)

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
        "vehicles_count": 50,
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [30, 45],  # Default, may be overridden by checkpoint
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": False,
    }

    # Override environment config with checkpoint configs if available
    if checkpoint_config is not None:
        print("Using config from checkpoint")
        # Extract training config for evaluation
        safety_bonus_coeff = checkpoint_config.get("safety_penalty", 0.01)
        print(f"Using safety_penalty from checkpoint: {safety_bonus_coeff}")
    else:
        print("Using config from command line flags")
        safety_bonus_coeff = FLAGS.config.get("safety_penalty", 0.01)
        print(f"Using safety_penalty from config: {safety_bonus_coeff}")

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

    # Create environment with potentially updated config
    env = gym.make("highway-v0", config=highway_config, render_mode=None)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    print(f"Final highway environment config:")
    for key, value in highway_config.items():
        if 'reward' in key or 'speed' in key:
            print(f"  {key}: {value}")

    # Run debug trajectory
    debug_single_trajectory(agent, env, FLAGS.max_steps, safety_bonus_coeff)


if __name__ == "__main__":
    app.run(main)