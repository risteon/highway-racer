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
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gym as old_gym  # Import old gym for spaces compatibility
import numpy as np
import threading
import queue
import time


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
flags.DEFINE_integer("eval_episodes", 16, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 500, "Logging interval.")
flags.DEFINE_integer("eval_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 128, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("replay_buffer_size", 30000, "Capacity of the replay buffer.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("record_video", False, "Record videos during training.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
flags.DEFINE_integer("save_buffer_interval", 50000, "Save buffer interval.")
flags.DEFINE_integer(
    "save_checkpoint_interval", 5000, "Steps between saving checkpoints."
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
flags.DEFINE_integer("num_envs", 8, "Number of parallel environments")

flags.DEFINE_integer("num_vehicles", 3, "Number of vehicles in the environment.")

config_flags.DEFINE_config_file(
    "config",
    "configs/highway_sac_config.py",
    "File path to the training hyperparameter configuration.",
)


class AsyncEnvStepper:
    """Handles asynchronous environment stepping with action queue.

    To be asychronous, split the envs into two sets, and update them alternatingly.
    """

    def __init__(self, envs, replay_buffer, log_interval, seed):

        # tuple (envsA, envsB) for vectorized environments
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.log_interval = log_interval
        self.seed = seed

        # Queues for communication between threads
        action_queue_size = 1
        self.action_queue = queue.Queue(maxsize=action_queue_size)
        self.result_queue = queue.Queue(maxsize=action_queue_size)

        # Thread control
        self.stop_event = threading.Event()
        self.env_thread = None

        # Current state
        self.current_observations = [None, None]
        self.current_infos = [None, None]

        # Statistics tracking
        self.total_steps = 0
        self.collision_counter = 0
        self.speed_ema = 0.0
        self.collision_ema = 0.0
        self.ema_beta = 3e-4
        self.offroad_termination_counter = 0

    def start(self):
        """Start the async environment stepping thread."""
        self.env_thread = threading.Thread(target=self._env_step_worker, daemon=True)
        self.env_thread.start()

    def stop(self):
        """Stop the async environment stepping thread."""
        self.stop_event.set()
        if self.env_thread and self.env_thread.is_alive():
            self.env_thread.join(timeout=5.0)

    def queue_actions(self, actions, step_num, index):
        """Queue actions for the next environment step."""
        try:
            self.action_queue.put((actions, step_num, index), timeout=None)
            return True
        except queue.Full:
            return False  # Queue is full, skip this step

    def get_results(self, timeout=None):
        """Get results from completed environment steps."""
        results = self.result_queue.get(timeout=timeout)
        return results

    def _env_step_worker(self):
        """Worker thread that processes environment steps."""
        # start with initial resets for both envs
        self.current_observations[0], self.current_infos[0] = self.envs[0].reset(
            seed=self.seed
        )
        self.result_queue.put((self.current_observations[0], 0))
        self.current_observations[1], self.current_infos[1] = self.envs[1].reset(
            seed=self.seed
        )
        self.result_queue.put((self.current_observations[1], 1))

        while not self.stop_event.is_set():
            try:
                # Get next action from queue
                actions, step_num, index = self.action_queue.get(timeout=None)

                # Step environment
                next_observations = self._step_environment(actions, step_num, index)

                # Put results in result queue
                self.result_queue.put((next_observations, index))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in env step worker: {e}")
                break

    def _step_environment(self, actions, step_num, index):
        """Execute a single environment step and process results."""
        # print(f"Stepping environment at step {step_num}")
        # Step the vectorized environment
        next_observations, rewards, terminations, truncations, infos = self.envs[
            index
        ].step(actions)
        # Handle episode terminations and create masks
        dones = terminations | truncations
        masks = 1.0 - dones.astype(float)

        if np.any(dones):
            # Handle `final_observation` in case of a truncation or termination
            real_next_observations = next_observations.copy()
            for idx in np.where(dones)[0]:
                real_next_observations[idx] = infos["final_obs"][idx]

            episode_infos = infos["final_info"]["episode"]
            r = episode_infos["r"][dones].mean()
            l = episode_infos["l"][dones].mean()
            t = episode_infos["t"][dones].mean()
            wandb.log(
                {
                    "training/return": r,
                    "training/length": l,
                    "training/time": t,
                },
                step=step_num,
            )

            ego_speeds = infos["speed"].copy()
            ego_speeds[dones] = infos["final_info"]["speed"][dones]
            # all that are not done are not crashed anyway
            collision_indicators = infos["final_info"]["crashed"].astype(float)

            offroad_terminations = (
                infos["final_info"]["rewards"]["on_road_reward"][dones] == 0.0
            )
            self.offroad_termination_counter += np.sum(offroad_terminations)

        else:
            real_next_observations = next_observations
            ego_speeds = infos["speed"]
            collision_indicators = infos["crashed"].astype(float)

        # Update EMAs
        self.speed_ema = (1 - self.ema_beta) * self.speed_ema + self.ema_beta * np.mean(
            ego_speeds
        )
        self.collision_ema = (
            1 - self.ema_beta
        ) * self.collision_ema + self.ema_beta * np.mean(collision_indicators)
        self.collision_counter += np.sum(collision_indicators)

        # Log EMAs periodically
        if step_num % self.log_interval == 0:
            wandb.log(
                {
                    "speed_ema": self.speed_ema,
                    "collision_ema": self.collision_ema,
                    "collision_counter": self.collision_counter,
                    "offroad_terminations": self.offroad_termination_counter,
                },
                step=step_num,
            )

        # Add experiences to replay buffer
        for env_idx in range(self.envs[index].num_envs):

            done = dones[env_idx]

            self.replay_buffer.insert(
                dict(
                    observations=self.current_observations[index][env_idx],
                    actions=actions[env_idx],
                    rewards=rewards[env_idx],
                    masks=masks[env_idx],
                    dones=done,
                    next_observations=real_next_observations[env_idx],
                )
            )

        # Update current state for next step
        self.current_observations[index] = next_observations
        self.current_infos[index] = infos
        self.total_steps += 1

        return next_observations


def make_env(env_name, highway_config, seed, idx):
    """Create a single environment for vectorization."""

    def thunk():
        env = gym.make(env_name, config=highway_config, render_mode=None)
        env = RecordEpisodeStatistics(env)
        env = FlattenObservation(env)  # Flatten (15, 6) -> (90,)
        env.action_space.seed(seed + idx)
        return env

    return thunk


def run_trajectory(
    agent,
    env,
    max_steps=1000,
    video: bool = False,
    output_range: Tuple[float, float] = None,
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

        # Use environment reward as-is (no safety reward modification)
        episode_return += reward
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
):
    episode_returns = []
    episode_lengths = []

    for i in range(num_episodes):
        images, episode_return, episode_length = run_trajectory(
            agent,
            eval_env,
            video=False,
            output_range=output_range,
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
            "vehicles_count": 8,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        "vehicles_count": FLAGS.num_vehicles,
        "vehicles_density": 0.75,
        "duration": 40,  # seconds
        "initial_spacing": 2,
        "collision_reward": -5.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 1.0,
        "lane_change_reward": 0.0,
        "reward_speed_range": [10, 40],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "normalize_reward": False,
        "offroad_terminal": True,
    }

    # Create vectorized environments. 2 batches!
    envsA = AsyncVectorEnv(
        [
            make_env(FLAGS.env_name, highway_config, FLAGS.seed, i)
            for i in range(0, FLAGS.num_envs)
        ],
        autoreset_mode="SameStep",
    )
    envsB = AsyncVectorEnv(
        [
            make_env(FLAGS.env_name, highway_config, FLAGS.seed, i)
            for i in range(FLAGS.num_envs, FLAGS.num_envs * 2)
        ],
        autoreset_mode="SameStep",
    )

    # Create single evaluation environment
    eval_env = gym.make(FLAGS.env_name, config=highway_config, render_mode="rgb_array")
    eval_env = FlattenObservation(eval_env)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")

    kwargs.pop("group_name", None)  # Remove group_name before passing to agent
    kwargs.pop("safety_penalty", None)  # Remove safety_penalty (not used)
    kwargs.pop("max_offroad_steps", None)  # Remove max_offroad_steps

    # agent = globals()[model_cls].create(
    #     FLAGS.seed, envsA.single_observation_space, envsA.single_action_space, **kwargs
    # )
    # use full vectorized obs space
    agent = globals()[model_cls].create(
        FLAGS.seed, envsA.observation_space, envsA.action_space, **kwargs
    )

    wandb_group_name = f"{FLAGS.config.group_name}"

    expert_replay_buffer = None
    if FLAGS.expert_replay_buffer:
        with open(FLAGS.expert_replay_buffer, "rb") as f:
            expert_replay_buffer = pickle.load(f)

    replay_buffer_size = FLAGS.replay_buffer_size

    replay_buffer = ReplayBuffer(
        envsA.single_observation_space,
        envsA.single_action_space,
        replay_buffer_size,
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

    action_min: np.ndarray = envsA.single_action_space.low
    action_max: np.ndarray = envsA.single_action_space.high

    if FLAGS.ramp_action:
        action_max[1] = max_action_schedule(0)

    pbar = tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
        dynamic_ncols=True,
    )

    # Initialize environment and async stepper

    # Create async environment stepper
    async_stepper = AsyncEnvStepper(
        (envsA, envsB),
        replay_buffer,
        log_interval=FLAGS.log_interval,
        seed=FLAGS.seed,
    )
    async_stepper.start()
    observations, index = async_stepper.get_results(timeout=None)

    # first action so that env can run in the background (always lagging behind one action)
    # async_stepper.queue_actions(envs.action_space.sample(), 0)

    try:
        for i in pbar:
            output_range = (action_min, action_max)

            if i < FLAGS.start_training:
                # Random actions for initial exploration
                # actions = np.array(
                #     [envs.single_action_space.sample() for _ in range(FLAGS.num_envs)]
                # )
                actions = envsA.action_space.sample()
            else:
                # Sample actions from agent using current observations
                actions, agent = agent.sample_actions(
                    observations, output_range=output_range
                )

            if FLAGS.ramp_action == "linear":
                action_max[1] = max_action_schedule(i)

            # Queue actions for async environment stepping
            if async_stepper.queue_actions(actions, i, index):
                # Actions queued successfully
                pass
            else:
                # Queue full, skip this step or handle differently
                print(f"Warning: Action queue full at step {i}")

            observations, index = async_stepper.get_results(timeout=None)

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
                    batch,
                    utd_ratio=FLAGS.utd_ratio,
                    output_range=mini_batch_output_range,
                )

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f"training/{k}": v}, step=i)

                    wandb.log(
                        {
                            "training/action_acc": np.mean(actions[:, 0]),
                            "training/action_steer": np.mean(actions[:, 1]),
                        },
                        step=i,
                    )

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
                                "highway_env_config": highway_config,  # Save environment configuration
                            },
                            f,
                        )
                    print(f"Saved checkpoint at step {i} to {policy_folder}")
                except Exception as e:
                    print(f"Cannot save checkpoints: {e}")

            pbar.set_description(
                f"Step {i}, Speed EMA: {async_stepper.speed_ema:.2f}, Collision EMA: {async_stepper.collision_ema:.3f}, Buffer: {len(replay_buffer)}"
            )

            if FLAGS.save_buffer and i % FLAGS.save_buffer_interval == 0:
                with open(os.path.join(FLAGS.save_dir, f"buffer_{i}.pkl"), "wb") as f:
                    pickle.dump(replay_buffer, f)

                # if i % FLAGS.log_interval == 0:
                #     wandb.log(
                #         {
                #             "training/action_queue_size": async_stepper.action_queue.qsize(),
                #             "training/result_queue_size": async_stepper.result_queue.qsize(),
                #         },
                #         step=i,
                #     )

    finally:
        # Stop async stepper and clean up
        print("Stopping async environment stepper...")
        async_stepper.stop()

        # Process any remaining results
        try:
            while True:
                remaining_result = async_stepper.get_results(timeout=0.1)
                print("Processed remaining environment result")
        except:
            print("Finished processing remaining environment results")

    if FLAGS.save_buffer:
        with open(os.path.join(FLAGS.save_dir, "final_buffer.pkl"), "wb") as f:
            pickle.dump(replay_buffer, f)

    envs.close()


if __name__ == "__main__":
    app.run(main)
