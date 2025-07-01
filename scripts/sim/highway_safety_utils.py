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
    """Calculate collision risk based on proximity to other vehicles."""
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
        collision_risk = -np.exp(
            -(min_distance - collision_threshold)
            / (safety_threshold - collision_threshold)
        )
    else:
        collision_risk = 0.0  # Safe distance

    return collision_risk


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
    if hasattr(env, 'vehicle') and env.vehicle is not None:
        vehicle = env.vehicle
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'vehicle'):
        vehicle = env.unwrapped.vehicle
    
    if vehicle is None:
        return 0.0
    
    # Method 1: Enhanced lateral distance detection (primary)
    # Access road through unwrapped environment if needed
    road = None
    if hasattr(env, 'road'):
        road = env.road
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'road'):
        road = env.unwrapped.road
    
    if road and hasattr(road, 'network'):
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
                if abs(lat) > lane_width/2:
                    return -0.5  # Fixed penalty for being offroad
        except:
            pass
    
    # Method 2: Fallback to built-in on_road property (less strict)
    if hasattr(vehicle, 'on_road'):
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
    if hasattr(env, 'vehicle') and env.vehicle is not None:
        vehicle = env.vehicle
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'vehicle'):
        vehicle = env.unwrapped.vehicle
    
    if vehicle is None:
        return False
    
    # Method 1: Check lateral distance from lane center (primary method)
    # Access road through unwrapped environment if needed
    road = None
    if hasattr(env, 'road'):
        road = env.road
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'road'):
        road = env.unwrapped.road
    
    if road and hasattr(road, 'network'):
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
                if abs(lat) > lane_width/2:
                    return True
        except:
            pass
    
    # Method 2: Fallback to built-in on_road property (less strict)
    if hasattr(vehicle, 'on_road'):
        if not vehicle.on_road:
            return True
    
    return False  # Assume on road if unable to determine


def debug_vehicle_position(env):
    """
    Debug vehicle road position with detailed information.
    
    Useful for understanding vehicle positioning relative to lane boundaries
    and debugging offroad detection issues.
    """
    if not hasattr(env, 'vehicle') or env.vehicle is None:
        # Try unwrapped environment
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'vehicle'):
            vehicle = env.unwrapped.vehicle
        else:
            return {"error": "No vehicle"}
    else:
        vehicle = env.vehicle
    
    debug_info = {
        'position': vehicle.position.tolist() if hasattr(vehicle.position, 'tolist') else list(vehicle.position),
        'on_road_property': getattr(vehicle, 'on_road', 'N/A'),
    }
    
    # Try to get detailed lane information
    # Access road through unwrapped environment if needed
    road = None
    if hasattr(env, 'road'):
        road = env.road
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'road'):
        road = env.unwrapped.road
    
    if road and hasattr(road, 'network'):
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
                
                debug_info.update({
                    'closest_lane': str(lane_index),
                    'distance_to_lane': float(distance),
                    'longitudinal': float(long),
                    'lateral': float(lat),
                    'lane_width': float(lane_width),
                    'lateral_margin': float(abs(lat) - lane_width/2),
                    'on_lane_strict': bool(lane.on_lane(vehicle.position, margin=0)),
                    'on_lane_default': bool(lane.on_lane(vehicle.position)),
                })
                
                # Alternative offroad detection based on lateral distance
                debug_info['offroad_by_lateral'] = abs(lat) > lane_width/2
        except Exception as e:
            debug_info['lane_error'] = str(e)
    
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
    if env is None:
        return 0.0
    
    # Access vehicle through unwrapped environment if needed
    vehicle = None
    if hasattr(env, 'vehicle') and env.vehicle is not None:
        vehicle = env.vehicle
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'vehicle'):
        vehicle = env.unwrapped.vehicle
    
    if vehicle is None:
        return 0.0
    
    # Calculate forward speed (speed in heading direction)
    import numpy as np
    forward_speed = vehicle.speed * np.cos(vehicle.heading)
    
    # Only reward forward motion (forward_speed > 0)
    if forward_speed <= 0:
        return 0.0  # No reward for backward/sideways driving
    
    # Linear mapping of forward speed to reward range [0, 1]
    min_speed, max_speed = reward_speed_range
    if forward_speed < min_speed:
        # Below optimal range: linear scaling from 0
        speed_reward = forward_speed / min_speed * 0.5  # Half reward below range
    elif forward_speed <= max_speed:
        # In optimal range: linear scaling from 0.5 to 1.0
        progress = (forward_speed - min_speed) / (max_speed - min_speed)
        speed_reward = 0.5 + progress * 0.5
    else:
        # Above optimal range: maximum reward but don't encourage excessive speed
        speed_reward = 1.0
    
    return float(np.clip(speed_reward, 0.0, 1.0))