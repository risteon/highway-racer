#!/usr/bin/env python3
"""
Debug script to understand highway-env offroad detection.
"""
import numpy as np
import gymnasium as gym
import highway_env

def debug_highway_offroad():
    """Test highway-env offroad detection with different configurations."""
    
    # Test configuration without offroad_terminal first
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
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": False,  # Start with False to avoid early termination
    }
    
    print("Creating highway environment...")
    env = gym.make("highway-v0", config=highway_config, render_mode=None)
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test some actions that might cause offroad behavior
    test_actions = [
        [0.0, 0.0],    # No steering, no acceleration
        [1.0, 0.0],    # Hard right, no acceleration  
        [-1.0, 0.0],   # Hard left, no acceleration
        [0.0, 1.0],    # No steering, full acceleration
        [1.0, 1.0],    # Hard right + acceleration
        [-1.0, 1.0],   # Hard left + acceleration
    ]
    
    for step, action in enumerate(test_actions):
        print(f"\n--- Step {step} ---")
        print(f"Action: {action}")
        
        # Get vehicle information before step
        # Try different ways to access the vehicle
        vehicle = None
        if hasattr(env, 'vehicle') and env.vehicle is not None:
            vehicle = env.vehicle
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'vehicle'):
            vehicle = env.unwrapped.vehicle
        
        if vehicle is not None:
            print(f"Vehicle position: {vehicle.position}")
            print(f"Vehicle on_road: {getattr(vehicle, 'on_road', 'N/A')}")
            
            # Get detailed lane information
            road = None
            if hasattr(env, 'road'):
                road = env.road
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'road'):
                road = env.unwrapped.road
                
            if road and hasattr(road, 'network'):
                try:
                    lane_index, distance = road.network.get_closest_lane_index(vehicle.position)
                    if lane_index is not None:
                        lane = road.network.get_lane(lane_index)
                        long, lat = lane.local_coordinates(vehicle.position)
                        lane_width = lane.width_at(long)
                        
                        print(f"Closest lane: {lane_index}")
                        print(f"Longitudinal: {long:.2f}, Lateral: {lat:.2f}")
                        print(f"Lane width: {lane_width:.2f}")
                        print(f"Lateral margin: {abs(lat) - lane_width/2:.2f}")
                        print(f"On lane (strict): {lane.on_lane(vehicle.position, margin=0)}")
                        print(f"On lane (default): {lane.on_lane(vehicle.position)}")
                        print(f"Offroad by lateral: {abs(lat) > lane_width/2}")
                except Exception as e:
                    print(f"Lane info error: {e}")
        else:
            print("Could not access vehicle object!")
            print(f"env.vehicle exists: {hasattr(env, 'vehicle')}")
            print(f"env.unwrapped: {hasattr(env, 'unwrapped')}")
            if hasattr(env, 'unwrapped'):
                print(f"env.unwrapped.vehicle exists: {hasattr(env.unwrapped, 'vehicle')}")
        
        # Take step
        obs, reward, done, truncated, info = env.step(np.array(action))
        print(f"Reward: {reward:.3f}, Done: {done}, Truncated: {truncated}")
        
        if done or truncated:
            print("Episode terminated!")
            break
    
    print("\n" + "="*50)
    print("Now testing with offroad_terminal=True...")
    
    # Test with offroad_terminal enabled
    highway_config["offroad_terminal"] = True
    env2 = gym.make("highway-v0", config=highway_config, render_mode=None)
    obs, info = env2.reset()
    
    print("Testing hard left turn with offroad_terminal=True...")
    for step in range(5):
        action = [-1.0, 0.5]  # Hard left + some acceleration
        print(f"\nStep {step}: action {action}")
        
        if hasattr(env2, 'vehicle') and env2.vehicle is not None:
            vehicle = env2.vehicle
            print(f"Position: {vehicle.position}, on_road: {getattr(vehicle, 'on_road', 'N/A')}")
        
        obs, reward, done, truncated, info = env2.step(np.array(action))
        print(f"Reward: {reward:.3f}, Done: {done}, Truncated: {truncated}")
        
        if done or truncated:
            print("Episode terminated!")
            if hasattr(env2, 'vehicle') and env2.vehicle is not None:
                print(f"Final on_road status: {getattr(env2.vehicle, 'on_road', 'N/A')}")
            break
    
    env.close()
    env2.close()

if __name__ == "__main__":
    debug_highway_offroad()