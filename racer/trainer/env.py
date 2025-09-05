from collections import namedtuple

from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
import highway_env  # This registers highway environments
from gymnasium.wrappers import (
    TimeLimit,
    RecordEpisodeStatistics,
    FlattenObservation,
    RecordVideo,
)


highway_config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 8,
        "features": ["presence", "x", "y", "vx", "vy", "heading"],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "lanes_count": 4,
    "vehicles_count": 10,
    "vehicles_density": 0.75,
    "duration": 40,  # seconds
    "initial_spacing": 2,
    "collision_reward": -10.0,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.5,
    "lane_change_reward": 0.0,
    "reward_speed_range": [10, 40],
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "normalize_reward": False,
    "offroad_terminal": True,
}


class EnvPair:
    def __init__(self, env_name, seed, num_envs: int) -> None:
        self.first = AsyncVectorEnv(
            [make_env(env_name, seed, i) for i in range(0, num_envs)],
            autoreset_mode="SameStep",
        )
        self.second = AsyncVectorEnv(
            [make_env(env_name, seed, i) for i in range(num_envs, num_envs * 2)],
            autoreset_mode="SameStep",
        )

    def close(self):
        self.first.close()
        self.second.close()

    def __getitem__(self, index):
        if index == 0:
            return self.first
        elif index == 1:
            return self.second
        else:
            raise IndexError("Invalid environment index")


def make_env(env_name, seed, idx, **kwargs):
    """Create a single environment for vectorization."""

    def thunk():
        env = gym.make(env_name, config=highway_config, **kwargs)
        env = RecordEpisodeStatistics(env)
        env = FlattenObservation(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk
