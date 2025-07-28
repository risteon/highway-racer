#!/usr/bin/env python
"""
Highway Trajectory Analyzer

This script analyzes highway environment trajectories using different agent types.
It can be used for:
1. Debugging reward functions with zero actions (straight driving)
2. Analyzing learned policy behavior
3. Comparing different agent types

The script uses a shared trajectory analysis framework that can work with
any agent implementing the AgentInterface.
"""
import warnings
import os
import numpy as np
from pathlib import Path
import gymnasium as gym
import highway_env
from gymnasium.wrappers import TimeLimit, FlattenObservation

# Import shared trajectory utilities
from highway_trajectory_utils import (
    run_highway_trajectory,
    ZeroActionAgent,
    LearnedPolicyAgent,
    AgentInterface,
)

# Import shared safety functions
from highway_safety_utils import (
    calculate_training_reward,
)

warnings.filterwarnings("ignore")


class ConstantActionAgent(AgentInterface):
    """Custom agent that drives with a constant action."""

    def __init__(self, action):
        self.action = np.array(action)

    def get_action(self, obs):
        """Return the constant action."""
        return self.action

    def reset(self, obs):
        """Reset the agent (no state to reset)."""
        pass


class ForwardStopBackwardAgent(AgentInterface):
    """Custom agent that drives forward, stops, then drives backward."""

    def __init__(self):
        self.step_count = 0

    def get_action(self, obs):
        """Get action based on current step: forward -> stop -> backward."""
        if self.step_count < 100:
            # Forward phase: moderate acceleration
            action = np.array([0.3, 0.0])  # [steering=0, acceleration=0.3]
        elif self.step_count < 120:
            # Stopping phase: strong deceleration
            action = np.array([-0.8, 0.0])  # [steering=0, acceleration=-0.8]
        else:
            # Backward phase: negative acceleration to go backward
            action = np.array([-0.5, 0.0])  # [steering=0, acceleration=-0.5]

        self.step_count += 1
        return action

    def reset(self, obs):
        """Reset step counter for new episode."""
        self.step_count = 0


def create_highway_environment(enable_rendering=False):
    """Create highway environment exactly as in training."""
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        # "vehicles_count": 50,
        "vehicles_count": 1,
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -10.0,
        "reward_speed_range": [20, 50],  # Training config range
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": True,  # Disable termination for this analysis
        # "collision_terminal": False,  # Disable collision termination
        "normalize_reward": False,
    }

    render_mode = "rgb_array" if enable_rendering else None
    env = gym.make("highway-v0", config=highway_config, render_mode=render_mode)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=200)  # Allow enough steps for full sequence

    return env, highway_config


def analyze_forward_stop_backward_trajectory(
    env, highway_config, safety_bonus_coeff=0.0, max_steps=1000, enable_video=False
):
    """
    Analyze trajectory with forward-stop-backward sequence to debug reward function behavior.

    Agent drives forward for 100 steps, decelerates to stop, then drives backward.
    This tests reward behavior across different driving scenarios and speeds.
    """
    print("=" * 80)
    print("HIGHWAY REWARD ANALYSIS - FORWARD-STOP-BACKWARD SEQUENCE")
    print("=" * 80)
    print(f"Agent behavior:")
    print(f"  Steps 0-99: Forward acceleration (action=[0.0, 0.3])")
    print(f"  Steps 100-119: Strong deceleration (action=[0.0, -0.8])")
    print(f"  Steps 120+: Backward acceleration (action=[0.0, -0.5])")
    print(f"Environment config:")
    for key, value in highway_config.items():
        if "reward" in key or "speed" in key or "collision" in key:
            print(f"  {key}: {value}")
    print(f"Safety bonus coefficient: {safety_bonus_coeff}")
    print()

    # Create forward-stop-backward agent
    # agent = ForwardStopBackwardAgent()
    agent = ConstantActionAgent(action=[0.0, 0.2])

    # Set video output path
    video_path = None
    if enable_video:
        video_path = "./evaluation/debug_videos/forward_stop_backward_analysis.mp4"

    # Run trajectory with debug analysis
    trajectory_metrics, images = run_highway_trajectory(
        agent=agent,
        env=env,
        highway_config=highway_config,
        max_steps=max_steps,
        render_video=enable_video,
        safety_bonus_coeff=safety_bonus_coeff,
        video_output_path=video_path,
        analysis_mode="debug",
    )

    # Extract step rewards for detailed analysis
    step_rewards = trajectory_metrics["step_rewards"]

    print("\n" + "=" * 80)
    print("REWARD ANALYSIS SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"Total steps: {len(step_rewards)}")
    print(f"Final cumulative env return: {trajectory_metrics['episode_return']:.3f}")
    print(
        f"Final cumulative training return: {trajectory_metrics['training_return']:.3f}"
    )
    print(
        f"Average env reward per step: {trajectory_metrics['episode_return']/len(step_rewards):.3f}"
    )
    print(
        f"Average training reward per step: {trajectory_metrics['training_return']/len(step_rewards):.3f}"
    )

    # Speed analysis
    speeds = [r["speed"] for r in step_rewards]
    forward_speeds = [r["forward_speed"] for r in step_rewards]
    print(f"\nSpeed Analysis:")
    print(f"  Average speed: {np.mean(speeds):.2f} m/s")
    print(f"  Speed range: [{np.min(speeds):.2f}, {np.max(speeds):.2f}] m/s")
    print(f"  Average forward speed: {np.mean(forward_speeds):.2f} m/s")
    print(
        f"  Forward speed range: [{np.min(forward_speeds):.2f}, {np.max(forward_speeds):.2f}] m/s"
    )
    print(f"  Training speed range: {highway_config['reward_speed_range']} m/s")

    # Lane analysis
    lanes = [r["lane"] for r in step_rewards if r["lane"] >= 0]
    if lanes:
        print(f"\nLane Analysis:")
        print(f"  Lanes occupied: {sorted(set(lanes))}")
        print(f"  Most common lane: {max(set(lanes), key=lanes.count)}")

    # Reward component analysis
    env_rewards = [r["env_reward"] for r in step_rewards]
    training_rewards = [r["training_reward"] for r in step_rewards]

    print(f"\nReward Component Analysis:")
    print(f"  Environment reward:")
    print(f"    Mean: {np.mean(env_rewards):.3f}, Std: {np.std(env_rewards):.3f}")
    print(f"    Range: [{np.min(env_rewards):.3f}, {np.max(env_rewards):.3f}]")
    print(f"  Training reward:")
    print(
        f"    Mean: {np.mean(training_rewards):.3f}, Std: {np.std(training_rewards):.3f}"
    )
    print(
        f"    Range: [{np.min(training_rewards):.3f}, {np.max(training_rewards):.3f}]"
    )

    # Check for specific issues
    print(f"\nIssue Detection:")

    # Check speed reward behavior
    speed_rewards = []
    for r in step_rewards:
        speed_component = r["components"].get("forward_speed_reward", 0.0)
        speed_rewards.append(speed_component)

    zero_speed_rewards = sum(1 for sr in speed_rewards if sr == 0.0)
    print(
        f"  Steps with zero speed reward: {zero_speed_rewards}/{len(step_rewards)} ({zero_speed_rewards/len(step_rewards)*100:.1f}%)"
    )

    if zero_speed_rewards == len(step_rewards):
        avg_speed = np.mean(speeds)
        speed_range = highway_config["reward_speed_range"]
        print(f"  ⚠️  ALL speed rewards are zero!")
        print(f"  ⚠️  Average speed {avg_speed:.1f} vs reward range {speed_range}")
        if avg_speed < speed_range[0]:
            print(f"  ⚠️  Speed too low: {avg_speed:.1f} < {speed_range[0]}")
        elif avg_speed > speed_range[1]:
            print(f"  ⚠️  Speed too high: {avg_speed:.1f} > {speed_range[1]}")

    # Check reward consistency
    reward_increases = sum(1 for r in env_rewards if r > 0)
    print(
        f"  Steps with positive env reward: {reward_increases}/{len(step_rewards)} ({reward_increases/len(step_rewards)*100:.1f}%)"
    )

    if all(r > 0 for r in env_rewards):
        print(
            f"  ⚠️  All environment rewards are positive - may indicate broken reward function"
        )

    return trajectory_metrics


def analyze_learned_policy_trajectory(
    policy_agent,
    env,
    highway_config,
    safety_bonus_coeff=0.01,
    max_steps=100,
    enable_video=False,
):
    """
    Analyze trajectory with a learned policy agent.

    This shows how the trained agent behaves and what rewards it receives.
    """
    print("=" * 80)
    print("HIGHWAY TRAJECTORY ANALYSIS - LEARNED POLICY")
    print("=" * 80)
    print(f"Environment config:")
    for key, value in highway_config.items():
        if "reward" in key or "speed" in key or "collision" in key:
            print(f"  {key}: {value}")
    print(f"Safety bonus coefficient: {safety_bonus_coeff}")
    print()

    # Wrap learned policy in AgentInterface
    agent = LearnedPolicyAgent(policy_agent)

    # Set video output path
    video_path = None
    if enable_video:
        video_path = "./evaluation/policy_videos/learned_policy_analysis.mp4"

    # Run trajectory with evaluation analysis
    trajectory_metrics, images = run_highway_trajectory(
        agent=agent,
        env=env,
        highway_config=highway_config,
        max_steps=max_steps,
        render_video=enable_video,
        safety_bonus_coeff=safety_bonus_coeff,
        video_output_path=video_path,
        analysis_mode="evaluation",
    )

    print("\n" + "=" * 80)
    print("LEARNED POLICY ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"Episode return: {trajectory_metrics['episode_return']:.3f}")
    print(f"Training return: {trajectory_metrics['training_return']:.3f}")
    print(f"Episode length: {trajectory_metrics['episode_length']}")
    print(f"Average speed: {trajectory_metrics['avg_ego_speed']:.2f} m/s")
    print(f"Collision occurred: {trajectory_metrics['collision_occurred']}")
    print(f"Safety violations: {trajectory_metrics['safety_violations']}")
    print(f"Average minimum distance: {trajectory_metrics['avg_min_distance']:.2f}m")
    print(f"Offroad violations: {trajectory_metrics['offroad_violations']}")

    return trajectory_metrics


def main():
    """Main function to run trajectory analysis."""
    print("Creating highway environment...")
    env, highway_config = create_highway_environment(
        enable_rendering=True
    )  # Enable rendering for video

    print("Running forward-stop-backward analysis...")
    # Use same safety coefficient as in training config
    safety_bonus_coeff = 0.0  # Start with 0 to isolate base rewards

    trajectory_metrics = analyze_forward_stop_backward_trajectory(
        env, highway_config, safety_bonus_coeff, max_steps=200, enable_video=True
    )

    env.close()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Note: To analyze a learned policy, use:
    # policy_agent = load_your_policy_here()
    # analyze_learned_policy_trajectory(policy_agent, env, highway_config)


if __name__ == "__main__":
    main()
