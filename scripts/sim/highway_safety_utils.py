#!/usr/bin/env python3
"""
Shared safety utilities for highway environment training and evaluation.

This module provides consistent safety reward functions and offroad detection
for both training and evaluation scripts.
"""
import numpy as np


def safety_reward_fn(obs, env=None):
    """
    Enhanced safety reward for highway driving.
    Penalizes collision risks and offroad violations.

    Args:
        obs: Flattened highway environment observation (90,) = 15 vehicles × 6 features
        env: Highway environment instance (for offroad detection)
        info: Additional environment info (optional)

    Returns:
        safety_reward: Negative reward for collision risk and offroad violations
    """
    # Calculate collision risk (existing logic)
    collision_risk = _calculate_collision_risk(obs)

    # Calculate offroad penalty using enhanced detection
    offroad_penalty = _calculate_offroad_penalty(env)

    # Combined safety reward
    return collision_risk + offroad_penalty


def _calculate_collision_risk(obs):
    """
    Calculate collision risk based on Time-to-Collision (TTC) in forward direction.

    Enhanced safety function that:
    - Uses TTC for predictive collision assessment
    - Focuses on vehicles in forward driving direction
    - Applies dual constraints: TTC threshold AND distance < 1m
    - Provides progressive penalties based on collision urgency

    Args:
        obs: Flattened observation (90,) = 15 vehicles × 6 features
             Features: [presence, x, y, vx, vy, heading]

    Returns:
        collision_risk: Negative penalty for collision risk (0.0 to -1.0)
    """
    # Reshape flattened observation back to (15, 6)
    obs = obs.reshape(-1, 6)  # [presence, x, y, vx, vy, heading]

    # Filter for present vehicles (presence > 0.5)
    present_mask = obs[:, 0] > 0.5
    present_vehicles = obs[present_mask]

    if len(present_vehicles) <= 1:
        return 0.0  # Only ego vehicle or no vehicles, no collision risk

    ego_vehicle = present_vehicles[0]  # First present vehicle is ego
    other_vehicles = present_vehicles[1:]  # Rest are other vehicles

    # Extract ego vehicle state
    ego_pos = ego_vehicle[1:3]  # [x, y]
    ego_vel = ego_vehicle[3:5]  # [vx, vy]
    ego_heading = ego_vehicle[5]  # heading angle

    # Calculate ego forward direction vector
    ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    # Safety parameters
    critical_ttc = 1.0  # seconds - immediate danger
    warning_ttc = 2.0  # seconds - early warning
    proximity_threshold = 1.0  # meters - Euclidean distance limit
    forward_cone_angle = np.pi / 8  # 22.5 degrees (±11.25° from heading)

    # Vectorized extraction for other vehicles
    other_pos = other_vehicles[:, 1:3]  # shape (N, 2)
    other_vel = other_vehicles[:, 3:5]  # shape (N, 2)

    # Relative position and velocity
    relative_pos = other_pos - ego_pos  # (N, 2)
    relative_vel = other_vel - ego_vel  # (N, 2)
    euclidean_distance = np.linalg.norm(relative_pos, axis=1)  # (N,)

    # # Avoid division by zero for zero distance
    # nonzero_mask = euclidean_distance > 0
    # if not np.any(nonzero_mask):
    #     return 0.0

    # # Only consider vehicles with nonzero distance
    # rel_pos_nz = relative_pos[nonzero_mask]
    # rel_vel_nz = relative_vel[nonzero_mask]
    # dist_nz = euclidean_distance[nonzero_mask]

    # Unit vector to each vehicle
    to_vehicle_unit = relative_pos / euclidean_distance[:, None]  # (M, 2)
    forward_dot = np.dot(to_vehicle_unit, ego_forward)  # (M,)
    angle_to_vehicle = np.arccos(np.clip(forward_dot, -1.0, 1.0))  # (M,)

    # Vehicles in forward cone
    in_cone_mask = angle_to_vehicle <= forward_cone_angle
    if not np.any(in_cone_mask):
        return 0.0

    # Filter for vehicles in cone
    rel_pos_fc = relative_pos[in_cone_mask]
    rel_vel_fc = relative_vel[in_cone_mask]
    dist_fc = euclidean_distance[in_cone_mask]

    # Forward distance and approach velocity
    forward_distance = np.dot(rel_pos_fc, ego_forward)  # (K,)
    approach_velocity = np.dot(rel_vel_fc, ego_forward)  # (K,)

    # Only calculate TTC if vehicles are approaching and in forward direction
    approaching_mask = (approach_velocity > 0) & (forward_distance > 0)
    if not np.any(approaching_mask):
        return 0.0

    fd = forward_distance[approaching_mask]
    av = approach_velocity[approaching_mask]
    dist = dist_fc[approaching_mask]
    ttc = fd / av

    # Apply dual safety constraints: TTC AND proximity
    danger_mask = (ttc < warning_ttc) & (dist < proximity_threshold)
    if not np.any(danger_mask):
        return 0.0

    ttc_danger = ttc[danger_mask]

    # Compute collision risk for each dangerous vehicle
    collision_risk = np.zeros_like(ttc_danger)  # (L,)
    critical_mask = ttc_danger < critical_ttc
    collision_risk[critical_mask] = -1.0
    if np.any(~critical_mask):
        urgency_factor = (warning_ttc - ttc_danger[~critical_mask]) / (
            warning_ttc - critical_ttc
        )
        collision_risk[~critical_mask] = -np.exp(-3.0 * (1.0 - urgency_factor))

    # Track maximum risk across all vehicles (most negative value)
    max_collision_risk = np.min(collision_risk)
    return max_collision_risk


def _calculate_offroad_penalty(env):
    """
    Enhanced offroad penalty calculation using lateral distance from lane center.

    This function uses the more accurate lateral distance detection instead of
    relying solely on highway-env's permissive on_road property.
    """
    if env is None:
        return 0.0

    # Access vehicle through unwrapped environment if needed
    vehicle = None
    if hasattr(env, "vehicle") and env.vehicle is not None:
        vehicle = env.vehicle
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "vehicle"):
        vehicle = env.unwrapped.vehicle

    if vehicle is None:
        return 0.0

    # Method 1: Enhanced lateral distance detection (primary)
    # Access road through unwrapped environment if needed
    road = None
    if hasattr(env, "road"):
        road = env.road
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "road"):
        road = env.unwrapped.road

    if road and hasattr(road, "network"):
        try:
            # highway-env API returns different formats, handle both
            result = road.network.get_closest_lane_index(vehicle.position)
            if isinstance(result, tuple) and len(result) == 2:
                lane_index, distance = result
            else:
                lane_index = result

            if lane_index is not None:
                lane = road.network.get_lane(lane_index)
                long, lat = lane.local_coordinates(vehicle.position)
                lane_width = lane.width_at(long)

                # Enhanced offroad detection: outside lane boundaries
                if abs(lat) > lane_width / 2:
                    return -0.5  # Fixed penalty for being offroad
        except:
            pass

    # Method 2: Fallback to built-in on_road property (less strict)
    if hasattr(vehicle, "on_road"):
        if not vehicle.on_road:
            return -0.5  # Fixed penalty for being offroad

    return 0.0  # On road or unable to determine


def is_vehicle_offroad(env):
    """
    Enhanced offroad detection using lateral distance from lane center.

    Returns True if vehicle is outside lane boundaries based on lateral distance,
    which is more accurate than highway-env's default on_road property.
    """
    if env is None:
        return False

    # Access vehicle through unwrapped environment if needed
    vehicle = None
    if hasattr(env, "vehicle") and env.vehicle is not None:
        vehicle = env.vehicle
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "vehicle"):
        vehicle = env.unwrapped.vehicle

    if vehicle is None:
        return False

    # TODO(risteon) sure that this isn't better?
    if hasattr(vehicle, "on_road"):
        if not vehicle.on_road:
            return True
        return False

    # Method 1: Check lateral distance from lane center (primary method)
    # Access road through unwrapped environment if needed
    road = None
    if hasattr(env, "road"):
        road = env.road
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "road"):
        road = env.unwrapped.road

    if road and hasattr(road, "network"):
        try:
            # highway-env API returns different formats, handle both
            result = road.network.get_closest_lane_index(vehicle.position)
            if isinstance(result, tuple) and len(result) == 2:
                lane_index, distance = result
            else:
                lane_index = result

            if lane_index is not None:
                lane = road.network.get_lane(lane_index)
                long, lat = lane.local_coordinates(vehicle.position)
                lane_width = lane.width_at(long)

                # Strict offroad detection: outside lane boundaries
                if abs(lat) > lane_width / 2:
                    return True
        except:
            pass

    # Method 2: Fallback to built-in on_road property (less strict)
    if hasattr(vehicle, "on_road"):
        if not vehicle.on_road:
            return True

    return False  # Assume on road if unable to determine


def debug_vehicle_position(env):
    """
    Debug vehicle road position with detailed information.

    Useful for understanding vehicle positioning relative to lane boundaries
    and debugging offroad detection issues.
    """
    if not hasattr(env, "vehicle") or env.vehicle is None:
        # Try unwrapped environment
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "vehicle"):
            vehicle = env.unwrapped.vehicle
        else:
            return {"error": "No vehicle"}
    else:
        vehicle = env.vehicle

    debug_info = {
        "position": (
            vehicle.position.tolist()
            if hasattr(vehicle.position, "tolist")
            else list(vehicle.position)
        ),
        "on_road_property": getattr(vehicle, "on_road", "N/A"),
    }

    # Try to get detailed lane information
    # Access road through unwrapped environment if needed
    road = None
    if hasattr(env, "road"):
        road = env.road
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "road"):
        road = env.unwrapped.road

    if road and hasattr(road, "network"):
        try:
            # highway-env API returns different formats, handle both
            result = road.network.get_closest_lane_index(vehicle.position)
            if isinstance(result, tuple) and len(result) == 2:
                lane_index, distance = result
            else:
                lane_index = result
                distance = 0.0

            if lane_index is not None:
                lane = road.network.get_lane(lane_index)
                long, lat = lane.local_coordinates(vehicle.position)
                lane_width = lane.width_at(long)

                debug_info.update(
                    {
                        "closest_lane": str(lane_index),
                        "distance_to_lane": float(distance),
                        "longitudinal": float(long),
                        "lateral": float(lat),
                        "lane_width": float(lane_width),
                        "lateral_margin": float(abs(lat) - lane_width / 2),
                        "on_lane_strict": bool(
                            lane.on_lane(vehicle.position, margin=0)
                        ),
                        "on_lane_default": bool(lane.on_lane(vehicle.position)),
                    }
                )

                # Alternative offroad detection based on lateral distance
                debug_info["offroad_by_lateral"] = abs(lat) > lane_width / 2
        except Exception as e:
            debug_info["lane_error"] = str(e)

    return debug_info


def calculate_forward_speed_reward(env, reward_speed_range=[20, 50]):
    """
    Calculate speed reward that only rewards forward driving.

    Fixes highway-env's backward driving reward bug by ensuring only
    forward motion in the desired speed range is rewarded.

    Args:
        env: Highway environment instance
        reward_speed_range: [min_speed, max_speed] for optimal reward

    Returns:
        speed_reward: Reward for forward driving speed (0.0 if backward)
    """
    # Access vehicle through unwrapped environment if needed
    vehicle = None
    if hasattr(env, "vehicle") and env.vehicle is not None:
        vehicle = env.vehicle
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "vehicle"):
        vehicle = env.unwrapped.vehicle

    if vehicle is None:
        return 0.0

    # Calculate forward speed (speed in heading direction)
    forward_speed = vehicle.velocity[0]
    min_speed, max_speed = reward_speed_range
    speed_reward = (forward_speed - min_speed) / (max_speed - min_speed)

    return float(np.clip(speed_reward, 0.0, 1.0))


def calculate_training_reward(
    env,
    base_reward,
    info,
    safety_bonus_coeff=0.01,
    reward_speed_range=[20, 50],
    speed_weight=0.4,
    next_obs=None,
):
    """
    Calculate the complete training reward used in RACER highway training.

    This function combines:
    1. Base environment reward
    2. Forward-only speed reward (fixes highway-env backward driving bug)
    3. Safety reward (collision risk + offroad penalties)

    Args:
        env: Highway environment instance
        base_reward: Raw reward from highway-env step
        info: Environment info dict containing reward components
        safety_bonus_coeff: Weight for safety reward component (default: 0.01)
        reward_speed_range: [min_speed, max_speed] for speed reward (default: [30, 45])
        speed_weight: Weight for speed component in highway-env (default: 0.4)

    Returns:
        training_reward: Complete reward used in training
        components: Dict with reward breakdown for logging/analysis
    """
    # Fix highway-env's backward driving reward bug
    # Replace highway-env's speed reward with forward-only speed reward
    # original_speed_reward = info["rewards"]["high_speed_reward"]
    # TODO(risteon) we don't need a fix!
    # forward_speed_reward = calculate_forward_speed_reward(env, reward_speed_range)

    # Adjust reward: remove original speed component and add forward-only version
    # Highway-env uses 0.4 weight for speed reward in default config
    # corrected_reward = (
    #     base_reward
    #     - original_speed_reward * speed_weight
    #     + forward_speed_reward * speed_weight
    # )

    # Compute safety reward (collision risk + offroad penalties)
    if next_obs is not None:
        safety_reward = safety_reward_fn(next_obs, env)
    else:
        # Fallback to offroad penalty only if observation not available
        safety_reward = _calculate_offroad_penalty(env)

    # Final training reward
    training_reward = base_reward + safety_reward * safety_bonus_coeff

    # Return breakdown for analysis
    components = {
        "base_reward": float(base_reward),
        # "corrected_reward": float(corrected_reward),
        # "original_speed_reward": float(original_speed_reward),
        # "forward_speed_reward": float(forward_speed_reward),
        "safety_reward": float(safety_reward),
        "training_reward": float(training_reward),
    }

    return training_reward, components
