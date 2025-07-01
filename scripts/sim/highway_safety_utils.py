#!/usr/bin/env python3
"""
Shared safety utilities for highway environment training and evaluation.

This module provides consistent safety reward functions and offroad detection
for both training and evaluation scripts.
"""
import numpy as np


def safety_reward_fn(obs, env=None, info=None):
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
    import numpy as np

    # Reshape flattened observation back to (15, 6)
    obs = obs.reshape(15, 6)  # [presence, x, y, vx, vy, heading]

    # Filter for present vehicles (presence > 0.5)
    present_vehicles = obs[obs[:, 0] > 0.5]

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
    warning_ttc = 3.0  # seconds - early warning
    proximity_threshold = 1.0  # meters - Euclidean distance limit
    forward_cone_angle = np.pi / 8  # 22.5 degrees (±11.25° from heading)

    max_collision_risk = 0.0  # Track the maximum risk from all vehicles

    for other_vehicle in other_vehicles:
        other_pos = other_vehicle[1:3]  # [x, y]
        other_vel = other_vehicle[3:5]  # [vx, vy]

        # Calculate relative vectors
        relative_pos = other_pos - ego_pos
        relative_vel = other_vel - ego_vel
        euclidean_distance = np.linalg.norm(relative_pos)

        # Check if vehicle is in forward cone
        if euclidean_distance > 0:  # Avoid division by zero
            to_vehicle_unit = relative_pos / euclidean_distance
            forward_dot = np.dot(to_vehicle_unit, ego_forward)
            angle_to_vehicle = np.arccos(np.clip(forward_dot, -1.0, 1.0))

            # Only consider vehicles in forward cone
            if angle_to_vehicle <= forward_cone_angle:
                # Calculate TTC in forward direction
                forward_distance = np.dot(relative_pos, ego_forward)
                approach_velocity = np.dot(relative_vel, ego_forward)

                # Only calculate TTC if vehicles are approaching and in forward direction
                if approach_velocity > 0 and forward_distance > 0:
                    ttc = forward_distance / approach_velocity

                    # Apply dual safety constraints: TTC AND proximity
                    if ttc < warning_ttc and euclidean_distance < proximity_threshold:
                        if ttc < critical_ttc:
                            # Maximum penalty for imminent collision
                            collision_risk = -1.0
                        else:
                            # Progressive penalty based on TTC urgency
                            urgency_factor = (warning_ttc - ttc) / (
                                warning_ttc - critical_ttc
                            )
                            collision_risk = -np.exp(-3.0 * (1.0 - urgency_factor))

                        # Track maximum risk across all vehicles
                        max_collision_risk = min(max_collision_risk, collision_risk)

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


def calculate_forward_speed_reward(env, reward_speed_range=[30, 45]):
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
