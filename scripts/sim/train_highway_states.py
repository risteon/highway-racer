#! /usr/bin/env python
import os
import pickle

import collections
from typing import Tuple
import gymnasium as gym
import highway_env  # This registers highway environments
from gymnasium.wrappers import (
    TimeLimit,
    RecordEpisodeStatistics,
    FlattenObservation,
    RecordVideo,
)
import gym as old_gym  # Import old gym for spaces compatibility
import numpy as np


# class BoundObservationWrapper(gym.Wrapper):
#     """Wrapper to bound infinite observation spaces for replay buffer compatibility."""

#     def __init__(self, env, obs_low=-100.0, obs_high=100.0):
#         super().__init__(env)
#         # Bound the observation space using old gym for compatibility
#         self.observation_space = old_gym.spaces.Box(
#             low=obs_low,
#             high=obs_high,
#             shape=env.observation_space.shape,
#             dtype=env.observation_space.dtype,
#         )


import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
import numpy as np
import moviepy.editor
from PIL import Image, ImageDraw

from jax import numpy as jnp

from jaxrl5.agents import *
from jaxrl5.data import ReplayBuffer

# Highway-env stuff
import warnings

warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "highway-fast-v0", "Highway environment name.")
flags.DEFINE_string("wandb_project", "highway_racer", "Project for W&B")
flags.DEFINE_string("comment", "", "Comment for W&B")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string(
    "expert_replay_buffer", "", "(Optional) Expert replay buffer pickle file."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 32, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 250, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 128, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("replay_buffer_size", 100000, "Capacity of the replay buffer.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("record_video", False, "Record videos during training.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
flags.DEFINE_integer("save_buffer_interval", 50000, "Save buffer interval.")
flags.DEFINE_integer(
    "save_checkpoint_interval", 2500, "Steps between saving checkpoints."
)
flags.DEFINE_string("checkpoint_dir", "policies", "Directory to save checkpoints.")
flags.DEFINE_integer("keep_checkpoints", 10, "Number of checkpoints to keep.")
flags.DEFINE_integer("utd_ratio", 8, "Updates per data point")
flags.DEFINE_integer(
    "reset_interval",
    None,
    "Parameter reset interval, in network updates (= env steps * UTD)",
)
flags.DEFINE_enum(
    "ramp_action",
    None,
    ["linear", "step"],
    "Should the max action be ramped up?",
    required=False,
)
flags.DEFINE_float(
    "action_penalty_start", None, "Start value for ramping up action", required=False
)
flags.DEFINE_float("action_penalty_end", 0.0, "End value for ramping up action")
flags.DEFINE_boolean("reset_ensemble", False, "Reset one ensemble member at a time")
flags.DEFINE_string("group_name_suffix", None, "Group name suffix")
config_flags.DEFINE_config_file(
    "config",
    "configs/highway_distributional_config.py",
    "File path to the training hyperparameter configuration.",
)


# Import shared safety functions
from highway_safety_utils import (
    safety_reward_fn,
    calculate_forward_speed_reward,
    is_vehicle_offroad,
    calculate_training_reward,
)


def run_trajectory(
    agent,
    env,
    max_steps=1000,
    video: bool = False,
    output_range: Tuple[float, float] = None,
    safety_bonus_coeff=0.01,
):
    obs, _ = env.reset()
    if hasattr(agent, "env_reset"):
        agent = agent.env_reset(obs)

    images = []
    episode_return = 0
    episode_length = 0

    for _ in range(max_steps):
        if video:
            images.append(env.render())

        action, agent = agent.sample_actions(obs, output_range=output_range)
        next_obs, reward, done, truncated, info = env.step(action)

        # Use shared training reward calculation
        training_reward, _ = calculate_training_reward(
            env, reward, info, safety_bonus_coeff, next_obs=next_obs
        )
        episode_return += training_reward
        episode_length += 1

        obs = next_obs

        if done or truncated:
            break

    return images, episode_return, episode_length


def evaluate(
    agent,
    eval_env,
    num_episodes: int,
    output_range: Tuple[float, float] = None,
    safety_bonus_coeff=0.01,
):
    episode_returns = []
    episode_lengths = []

    for i in range(num_episodes):
        images, episode_return, episode_length = run_trajectory(
            agent,
            eval_env,
            video=False,
            output_range=output_range,
            safety_bonus_coeff=safety_bonus_coeff,
        )
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    wandb.log(
        {
            "evaluation/episode_returns_histogram": wandb.Histogram(episode_returns),
            "evaluation/episode_return": np.mean(episode_returns),
            "evaluation/episode_length_histogram": wandb.Histogram(episode_lengths),
            "evaluation/episode_length": np.mean(episode_lengths),
        }
    )


def max_action_schedule(i):
    return min(1, 3 * i / FLAGS.max_steps - 0.5)


def main(_):
    # Highway environment setup
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
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 50],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offroad_terminal": True,  # Enable proper offroad detection
    }

    # Set render mode if video recording is enabled
    # render_mode = "rgb_array" if FLAGS.record_video else None
    env = gym.make(FLAGS.env_name, config=highway_config, render_mode=None)

    # Apply RecordEpisodeStatistics FIRST to ensure episode info propagation
    env = RecordEpisodeStatistics(env)
    env = FlattenObservation(env)  # Flatten (15, 6) -> (90,)
    env = TimeLimit(env, max_episode_steps=1000)

    eval_env = gym.make(FLAGS.env_name, config=highway_config, render_mode="rgb_array")
    eval_env = FlattenObservation(eval_env)  # Flatten (15, 6) -> (90,)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")

    safety_bonus_coeff = kwargs.pop("safety_penalty", 0.0)
    kwargs.pop("group_name", None)  # Remove group_name before passing to agent
    kwargs.pop(
        "max_offroad_steps", None
    )  # Remove max_offroad_steps before passing to agent

    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    wandb_group_name = f"{FLAGS.config.group_name}"

    expert_replay_buffer = None
    if FLAGS.expert_replay_buffer:
        with open(FLAGS.expert_replay_buffer, "rb") as f:
            expert_replay_buffer = pickle.load(f)

    replay_buffer_size = FLAGS.replay_buffer_size

    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        replay_buffer_size,
        extra_fields=["safety"],
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
        }
    )
    if FLAGS.expert_replay_buffer:
        expert_replay_buffer_iterator = expert_replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
            }
        )

    observation, info = env.reset()

    wandb.init(
        project=FLAGS.wandb_project,
        notes=FLAGS.comment,
        group=(
            f"{wandb_group_name}-highway"
            if FLAGS.group_name_suffix is None
            else f"{wandb_group_name}-{FLAGS.group_name_suffix}-highway"
        ),
    )
    config_for_wandb = {
        **FLAGS.flag_values_dict(),
        "config": dict(FLAGS.config),
    }
    wandb.config.update(config_for_wandb)

    reset_interval = FLAGS.reset_interval
    if FLAGS.reset_ensemble:
        reset_interval = reset_interval // agent.num_qs

    action_min: np.ndarray = env.action_space.low
    action_max: np.ndarray = env.action_space.high

    speed_ema = 0.0
    safety_ema = 0.0
    collision_ema = 0.0

    # Offroad tracking
    # offroad_step_counter = 0
    # max_offroad_steps = FLAGS.config.get("max_offroad_steps", 20)
    offroad_terminations = 0
    # total collisions during training
    collision_counter = 0

    ema_beta = 3e-4

    if FLAGS.ramp_action:
        action_max[1] = max_action_schedule(0)

    pbar = tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
        dynamic_ncols=True,
    )
    for i in pbar:
        if hasattr(agent, "target_entropy") and hasattr(agent.target_entropy, "set"):
            agent = agent.replace(target_entropy=-env.action_space.shape[-1])

        output_range = (action_min, action_max)

        action, agent = agent.sample_actions(observation, output_range=output_range)
        if i < FLAGS.start_training:
            # Random action for initial exploration
            action = env.action_space.sample()
        else:
            action = np.clip(action, env.action_space.low, env.action_space.high)

        if FLAGS.ramp_action == "linear":
            action_max[1] = max_action_schedule(i)

        next_observation, reward, done, truncated, info = env.step(action)

        # Use shared training reward calculation
        training_reward, reward_components = calculate_training_reward(
            env,
            reward,
            info,
            safety_bonus_coeff,
            reward_speed_range=highway_config["reward_speed_range"],
            next_obs=next_observation,
        )
        reward = training_reward
        safety_bonus = reward_components["safety_reward"]

        # Check for collision from highway-env reward components
        collision_occurred = info["rewards"]["collision_reward"] < 0.0
        collision_indicator = 1.0 if collision_occurred else 0.0
        if collision_occurred:
            collision_counter += 1

        # TODO(risteon) using standard env termination
        # Check for offroad status and update counter
        # is_offroad = is_vehicle_offroad(env)
        # if is_offroad:
        #     offroad_step_counter += 1
        #     # Terminate episode if offroad for too long
        #     if offroad_step_counter >= max_offroad_steps:
        #         done = True
        #         offroad_terminations += 1
        # else:
        #     offroad_step_counter = 0  # Reset counter when back on road

        # Update EMAs for logging
        obs_reshaped = next_observation.reshape(15, 6)  # Reshape flattened obs
        present_vehicles = obs_reshaped[obs_reshaped[:, 0] > 0.5]
        if len(present_vehicles) > 0:
            ego_vehicle = present_vehicles[0]  # First present vehicle is ego
            # ego_speed = np.linalg.norm(ego_vehicle[3:5])  # vx, vy of ego vehicle
            ego_speed = ego_vehicle[3]  # vx of ego vehicle
        else:
            ego_speed = 0.0
        speed_ema = (1 - ema_beta) * speed_ema + ema_beta * ego_speed
        safety_ema = (1 - ema_beta) * safety_ema + ema_beta * safety_bonus
        collision_ema = (1 - ema_beta) * collision_ema + ema_beta * collision_indicator

        if not done or truncated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
                safety=-safety_bonus,
            )
        )

        observation = next_observation

        if done or truncated:
            if "episode" in info:
                for k, v in info["episode"].items():
                    decode = {"r": "return", "l": "length", "t": "time"}
                    wandb_log = {f"training/{decode[k]}": v}
                    wandb.log(wandb_log, step=i)

            is_offroad = is_vehicle_offroad(env)
            if is_offroad:
                offroad_terminations += 1
            observation, info = env.reset()

        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)

            if FLAGS.expert_replay_buffer:
                expert_batch = next(expert_replay_buffer_iterator)
                # Mix expert and online data
                for key in batch.keys():
                    batch[key] = np.concatenate(
                        [
                            batch[key][: FLAGS.batch_size // 2],
                            expert_batch[key][: FLAGS.batch_size // 2],
                        ],
                        axis=0,
                    )

            if FLAGS.ramp_action == "step":
                action_max[1] = max_action_schedule(i)

            output_range = (action_min, action_max)

            mini_batch_output_range = (
                jnp.tile(output_range[0], (FLAGS.batch_size * FLAGS.utd_ratio, 1)),
                jnp.tile(output_range[1], (FLAGS.batch_size * FLAGS.utd_ratio, 1)),
            )

            agent, update_info = agent.update(
                batch, utd_ratio=FLAGS.utd_ratio, output_range=mini_batch_output_range
            )

            if hasattr(agent, "update_safety"):
                agent, safety_info = agent.update_safety(-safety_ema)
                update_info = {**update_info, **safety_info}

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

            if (
                reset_interval is not None
                and i % reset_interval == 0
                and i < int(FLAGS.max_steps * 0.8)
            ):
                if FLAGS.reset_ensemble:
                    agent, ensemble_info = agent.reset_ensemble_member()
                    for k, v in ensemble_info.items():
                        wandb.log({f"reset/{k}": v}, step=i)
                else:
                    agent = agent.reset(exclude=["critic", "target_critic"])

        if i % FLAGS.eval_interval == 0:
            evaluate(
                agent,
                eval_env,
                FLAGS.eval_episodes,
                output_range=output_range,
                safety_bonus_coeff=safety_bonus_coeff,
            )

        # Save checkpoints at specified intervals
        if i % FLAGS.save_checkpoint_interval == 0 or i == 100:
            try:
                # Handle case where wandb.run.name might be None (offline mode)
                run_name = (
                    wandb.run.name
                    if wandb.run.name is not None
                    else f"highway_run_{FLAGS.seed}"
                )
                policy_folder = os.path.abspath(
                    os.path.join(FLAGS.checkpoint_dir, run_name)
                )
                os.makedirs(policy_folder, exist_ok=True)

                # Convert config to regular dict to avoid serialization issues
                config_dict = dict(
                    FLAGS.config
                )  # Use dict() constructor for ConfigDict

                # Convert flags to regular dict
                flags_dict = {}
                for key, value in FLAGS.flag_values_dict().items():
                    if key != "config":  # Skip config to avoid circular reference
                        try:
                            # Handle different types properly
                            if hasattr(value, "to_dict"):
                                flags_dict[key] = value.to_dict()
                            elif (
                                isinstance(value, (str, int, float, bool))
                                or value is None
                            ):
                                flags_dict[key] = value
                            else:
                                flags_dict[key] = str(
                                    value
                                )  # Convert complex types to string
                        except Exception as e:
                            print(f"Warning: Could not serialize flag {key}: {e}")
                            flags_dict[key] = str(value)

                param_dict = {
                    "actor": agent.actor,
                    "critic": agent.critic,
                    "target_critic_params": agent.target_critic,
                    "temp": agent.temp,
                    "rng": agent.rng,
                    "config": config_dict,  # Save training configuration as regular dict
                    "training_flags": flags_dict,  # Save all training flags as regular dict
                }

                # Config and flags prepared for saving
                if hasattr(agent, "limits"):
                    param_dict["limits"] = agent.limits
                if hasattr(agent, "q_entropy_lagrange"):
                    param_dict["q_entropy_lagrange"] = agent.q_entropy_lagrange

                # Save main model checkpoint using Orbax
                checkpoints.save_checkpoint(
                    policy_folder, param_dict, step=i, keep=FLAGS.keep_checkpoints
                )

                # Save config separately using pickle (more reliable for non-JAX data)
                import pickle

                config_file = os.path.join(policy_folder, f"config_{i}.pkl")
                with open(config_file, "wb") as f:
                    pickle.dump(
                        {
                            "config": config_dict, 
                            "training_flags": flags_dict,
                            "highway_env_config": highway_config  # Save environment configuration
                        }, f
                    )
                print(f"Saved checkpoint at step {i} to {policy_folder}")
            except Exception as e:
                print(f"Cannot save checkpoints: {e}")

        pbar.set_description(
            f"Step {i}, Return: {reward:.2f}, Speed EMA: {speed_ema:.2f}, Safety EMA: {safety_ema:.2f}, Collision EMA: {collision_ema:.3f}"
        )

        if FLAGS.save_buffer and i % FLAGS.save_buffer_interval == 0:
            with open(os.path.join(FLAGS.save_dir, f"buffer_{i}.pkl"), "wb") as f:
                pickle.dump(replay_buffer, f)

        wandb_log = {}
        if i % FLAGS.log_interval == 0:
            wandb_log["speed_ema"] = speed_ema
            wandb_log["safety_ema"] = safety_ema
            wandb_log["collision_ema"] = collision_ema
            wandb_log["offroad_terminations"] = offroad_terminations
            wandb_log["collision_counter"] = collision_counter
            if wandb_log:
                wandb.log(wandb_log, step=i)

    if FLAGS.save_buffer:
        with open(os.path.join(FLAGS.save_dir, "final_buffer.pkl"), "wb") as f:
            pickle.dump(replay_buffer, f)


if __name__ == "__main__":
    app.run(main)
