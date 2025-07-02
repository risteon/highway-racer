#!/usr/bin/env python3
"""
Shared trajectory analysis utilities for highway environment.

This module provides reusable functions for running and analyzing highway
trajectories with different types of agents (learned policies, dummy policies, etc.).
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from moviepy.editor import ImageSequenceClip
from abc import ABC, abstractmethod

# Import shared safety functions
from highway_safety_utils import (
    safety_reward_fn,
    is_vehicle_offroad,
    debug_vehicle_position,
    calculate_training_reward,
)


class AgentInterface(ABC):
    """Abstract interface for agents that can be used in trajectory analysis."""

    @abstractmethod
    def get_action(self, obs):
        """Get action from agent given observation."""
        pass

    @abstractmethod
    def reset(self, obs):
        """Reset agent state for new episode."""
        pass


class ZeroActionAgent(AgentInterface):
    """Dummy agent that always returns zero actions (straight driving)."""

    def get_action(self, obs):
        """Return zero actions for straight driving."""
        return np.array([0.0, 0.0])  # [steering=0, acceleration=0]

    def reset(self, obs):
        """No state to reset for zero action agent."""
        pass


class LearnedPolicyAgent(AgentInterface):
    """Wrapper for learned policies (SAC/DistributionalSAC agents)."""

    def __init__(self, policy_agent):
        """Initialize with a trained policy agent."""
        self.policy_agent = policy_agent

    def get_action(self, obs):
        """Sample action from learned policy."""
        action, self.policy_agent = self.policy_agent.sample_actions(obs)
        return action

    def reset(self, obs):
        """Reset policy agent for new episode."""
        if hasattr(self.policy_agent, "env_reset"):
            self.policy_agent = self.policy_agent.env_reset(obs)


def run_highway_trajectory(
    agent: AgentInterface,
    env,
    highway_config,
    max_steps=1000,
    render_video=False,
    safety_bonus_coeff=0.01,
    video_output_path=None,
    analysis_mode="evaluation",  # "evaluation" or "debug"
):
    """
    Run a single highway trajectory with any agent and collect detailed metrics.

    Args:
        agent: AgentInterface implementation (learned policy, zero actions, etc.)
        env: Highway environment instance
        highway_config: Environment configuration dict
        max_steps: Maximum steps per episode
        render_video: Whether to capture video frames
        safety_bonus_coeff: Safety reward coefficient for training reward calculation
        video_output_path: Path to save video (if None, auto-generated)
        analysis_mode: "evaluation" for concise metrics, "debug" for detailed analysis

    Returns:
        trajectory_metrics: Dict with episode metrics
        images: List of video frames (if render_video=True)
    """
    obs, _ = env.reset()
    agent.reset(obs)

    # Track metrics
    images = []
    episode_return = 0
    training_return = 0
    episode_length = 0
    safety_violations = 0
    min_distances = []
    ego_speeds = []
    collision_occurred = False
    offroad_violations = 0
    current_offroad_duration = 0
    offroad_durations = []
    step_rewards = []  # For detailed debug analysis

    # Print header for debug mode
    if analysis_mode == "debug":
        print(
            f"{'Step':<4} | {'Action':<15} | {'Env Rew':<8} | {'Train Rew':<9} | {'Speed':<6} | {'Lane':<4} | {'Components'}"
        )
        print("-" * 100)

    for step in range(max_steps):
        # Get action from agent
        action = agent.get_action(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Capture video frame if enabled
        if render_video:
            img = env.render()
            if img is not None:
                # Convert to PIL Image for text overlay
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)

                # Extract ego vehicle info for overlay
                obs_reshaped = obs.reshape(15, 6)
                present_vehicles = obs_reshaped[obs_reshaped[:, 0] > 0.5]
                if len(present_vehicles) > 0:
                    ego_vehicle = present_vehicles[0]
                    ego_speed = np.linalg.norm(ego_vehicle[3:5])  # vx, vy

                    # Calculate safety metrics for overlay
                    safety_reward = safety_reward_fn(obs, env)
                    is_offroad = is_vehicle_offroad(env)
                    debug_info = debug_vehicle_position(env)

                    # Grid layout parameters
                    row_height = 20
                    col_width = 200
                    start_x = 10
                    start_y = 10

                    # Define all info items to display
                    info_items = [
                        f"Step: {step}",
                        f"Action: [{action[0]:.1f}, {action[1]:.1f}]",
                        f"Speed: {ego_speed:.2f} m/s",
                        f"Safety: {safety_reward:.3f}",
                        f"Env Return: {episode_return:.2f}",
                        f"Training Return: {training_return:.2f}",
                        f"Offroad: {'YES' if is_offroad else 'NO'}",
                    ]

                    # Add lateral positioning info if available
                    if "lateral" in debug_info and "lane_width" in debug_info:
                        lateral = debug_info["lateral"]
                        lane_width = debug_info["lane_width"]
                        margin = lateral - lane_width / 2
                        info_items.extend(
                            [
                                f"Lateral: {lateral:.2f}m",
                                f"Lane Width: {lane_width:.2f}m",
                                f"Margin: {margin:.2f}m",
                            ]
                        )

                    # Add on_road property if available
                    if "on_road_property" in debug_info:
                        info_items.append(f"OnRoad: {debug_info['on_road_property']}")

                    # Arrange items in 3-row grid
                    for i, item in enumerate(info_items):
                        row = i % 3
                        col = i // 3
                        x = start_x + col * col_width
                        y = start_y + row * row_height

                        # Special color handling for certain items
                        text_color = (255, 255, 255)  # Default white
                        if "Offroad: YES" in item:
                            text_color = (255, 0, 0)  # Red for offroad
                        elif "Offroad: NO" in item:
                            text_color = (0, 255, 0)  # Green for on road
                        elif (
                            "Margin:" in item
                            and "lateral" in debug_info
                            and "lane_width" in debug_info
                        ):
                            # Red if outside lane boundaries, green if inside
                            if (
                                abs(debug_info["lateral"])
                                > debug_info["lane_width"] / 2
                            ):
                                text_color = (255, 0, 0)
                            else:
                                text_color = (0, 255, 0)

                        draw.text((x, y), item, fill=text_color)

                images.append(np.asarray(img_pil))

        # Take environment step
        next_obs, env_reward, done, truncated, info = env.step(action)

        if next_obs[3] > 39.6:
            print("x")

        # Calculate training reward using shared function
        training_reward, reward_components = calculate_training_reward(
            env,
            env_reward,
            info,
            safety_bonus_coeff,
            reward_speed_range=highway_config.get("reward_speed_range", [20, 50]),
            next_obs=next_obs,
        )
        safety_reward = reward_components["safety_reward"]

        # Extract vehicle information from observation for analysis
        obs_reshaped = obs.reshape(15, 6)
        present_vehicles = obs_reshaped[obs_reshaped[:, 0] > 0.5]

        if len(present_vehicles) > 0:
            ego_vehicle = present_vehicles[0]
            ego_pos = ego_vehicle[1:3]  # [x, y]
            ego_vel = ego_vehicle[3:5]  # [vx, vy]
            ego_speed = np.linalg.norm(ego_vel)
            ego_forward_speed = ego_vel[0]  # Only x-component

            # Determine current lane (approximate)
            ego_y = ego_pos[1]
            lane_width = 4.0  # Standard highway lane width
            current_lane = int(
                (ego_y + 2 * lane_width) / lane_width
            )  # Rough lane estimation

            ego_speeds.append(ego_speed)
        else:
            ego_speed = 0.0
            ego_forward_speed = 0.0
            current_lane = -1

        # Track safety violations
        if safety_reward < -0.5:  # Threshold for safety violation
            safety_violations += 1

        # Track offroad status
        is_offroad = is_vehicle_offroad(env)
        if is_offroad:
            offroad_violations += 1
            current_offroad_duration += 1
        else:
            if current_offroad_duration > 0:
                offroad_durations.append(current_offroad_duration)
                current_offroad_duration = 0

        # Track minimum distance to other vehicles
        if len(present_vehicles) > 1:
            other_vehicles = present_vehicles[1:]
            ego_pos = present_vehicles[0][1:3]
            other_pos = other_vehicles[:, 1:3]
            distances = np.linalg.norm(other_pos - ego_pos, axis=1)
            min_distances.append(np.min(distances))

        # Check for collision
        if info["rewards"]["collision_reward"] < 0.0:
            collision_occurred = True

        # Update returns
        episode_return += env_reward
        training_return += training_reward
        episode_length += 1

        # Store detailed step data for debug mode
        if analysis_mode == "debug":
            step_rewards.append(
                {
                    "step": step,
                    "env_reward": env_reward,
                    "training_reward": training_reward,
                    "speed": ego_speed,
                    "forward_speed": ego_forward_speed,
                    "lane": current_lane,
                    "components": reward_components,
                }
            )

            # Print step info
            action_str = f"[{action[0]:.1f}, {action[1]:.1f}]"
            components_str = f"C:{info['rewards']['collision_reward']:.0f} R:{info['rewards']['right_lane_reward']:.2f} S:{info['rewards']['high_speed_reward']:.2f}"
            print(
                f"{step:<4} | {action_str:<15} | {env_reward:<8.3f} | {training_reward:<9.3f} | {ego_speed:<6.1f} | {current_lane:<4} | {components_str}"
            )

        obs = next_obs

        if done or truncated:
            if analysis_mode == "debug":
                print(f"\nEpisode ended at step {step}")
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
    }

    # Add debug-specific data
    if analysis_mode == "debug":
        trajectory_metrics["step_rewards"] = step_rewards

    # Save video if requested and frames were captured
    if render_video and images:
        if video_output_path is None:
            video_output_dir = "./evaluation/trajectory_videos"
            Path(video_output_dir).mkdir(parents=True, exist_ok=True)
            video_output_path = (
                f"{video_output_dir}/highway_trajectory_{analysis_mode}.mp4"
            )
        else:
            Path(video_output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"\nSaving video with {len(images)} frames...")
            ImageSequenceClip(sequence=images, fps=15).write_videofile(
                video_output_path, verbose=False, logger=None
            )
            print(f"Video saved to: {video_output_path}")
            trajectory_metrics["video_path"] = video_output_path
        except Exception as e:
            print(f"Error saving video: {e}")

    return trajectory_metrics, images
