#! /usr/bin/env python
"""
Distributed SAC Training Script for Highway Environment

This script implements a distributed training system that parallelizes environment
interactions while maintaining a centralized learning loop for optimal sample
efficiency and training stability.

Architecture:
- Multiple environment workers running in parallel processes
- Centralized learner process handling all neural network updates
- Thread-safe replay buffer for experience collection
- Efficient policy parameter synchronization
"""
import os
import pickle
import collections
from typing import Tuple, Dict, List, Any
import multiprocessing as mp
import threading
import queue
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

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

import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
import numpy as np
import moviepy.editor
from PIL import Image, ImageDraw

from jax import numpy as jnp
import jax

from jaxrl5.agents import *
from jaxrl5.data import ReplayBuffer

# Highway-env stuff
import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

# Original flags
flags.DEFINE_string("env_name", "highway-fast-v0", "Highway environment name.")
flags.DEFINE_string("wandb_project", "highway_racer", "Project for W&B")
flags.DEFINE_string("comment", "", "Comment for W&B")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string(
    "expert_replay_buffer", "", "(Optional) Expert replay buffer pickle file."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 16, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 250, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size (increased for distributed).")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(2e3), "Number of training steps to start training."
)
flags.DEFINE_integer(
    "replay_buffer_size",
    100000,
    "Capacity of the replay buffer (increased for distributed).",
)
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
flags.DEFINE_integer(
    "utd_ratio", 16, "Updates per data point (increased for distributed)"
)
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

# New distributed training flags
flags.DEFINE_integer("num_workers", 4, "Number of environment workers")
flags.DEFINE_integer("worker_steps_per_iteration", 32, "Steps per worker iteration")
flags.DEFINE_float("policy_update_frequency", 1.0, "Policy broadcast frequency (Hz)")
flags.DEFINE_integer("experience_batch_size", 128, "Experiences per worker transfer")
flags.DEFINE_integer(
    "learner_update_ratio", 8, "Continuous updates (higher = more training per experience)"
)
flags.DEFINE_integer("max_queue_size", 100, "Maximum experience queue size")
flags.DEFINE_boolean(
    "debug_distributed", False, "Enable distributed training debug logging"
)

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


class DistributedReplayBuffer:
    """Thread-safe replay buffer for distributed experience collection."""

    def __init__(self, observation_space, action_space, capacity, extra_fields=None):
        self.buffer = ReplayBuffer(
            observation_space, action_space, capacity, extra_fields=extra_fields
        )
        self.lock = threading.Lock()
        self._total_inserted = 0

    def insert_batch(self, experiences: List[Dict[str, Any]]) -> None:
        """Thread-safe batch insertion of experiences."""
        with self.lock:
            for exp in experiences:
                self.buffer.insert(exp)
                self._total_inserted += 1

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Thread-safe sampling of experiences."""
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            return self.buffer.sample(batch_size)

    @property
    def size(self) -> int:
        """Thread-safe size access."""
        with self.lock:
            return len(self.buffer)

    @property
    def total_inserted(self) -> int:
        """Total number of experiences inserted."""
        return self._total_inserted

    def seed(self, seed: int) -> None:
        """Seed the buffer."""
        with self.lock:
            self.buffer.seed(seed)

    def get_iterator(self, queue_size: int = 2, sample_args: Dict[str, Any] = None):
        """Get iterator for sampling."""
        with self.lock:
            if sample_args is None:
                sample_args = {}
            return self.buffer.get_iterator(queue_size=queue_size, sample_args=sample_args)


def create_highway_env(
    highway_config: Dict[str, Any],
    env_name: str = "highway-fast-v0",
    render_mode: str = None,
    seed: int = None,
):
    """Create a highway environment with consistent configuration."""
    env = gym.make(env_name, config=highway_config, render_mode=render_mode)
    env = RecordEpisodeStatistics(env)
    env = FlattenObservation(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def serialize_policy_params(agent) -> bytes:
    """Serialize agent parameters for inter-process communication."""
    # Extract just the parameters we need for action sampling
    policy_params = {
        "actor_params": agent.actor.params,
        "rng": agent.rng,
    }

    # Add limits if available (for distributional SAC)
    if hasattr(agent, "limits"):
        policy_params["limits_params"] = agent.limits.params

    return pickle.dumps(policy_params)


def deserialize_policy_params(serialized_params: bytes) -> Dict[str, Any]:
    """Deserialize policy parameters."""
    return pickle.loads(serialized_params)


class EnvironmentWorker:
    """Individual worker process for environment interaction."""

    def __init__(
        self,
        worker_id: int,
        highway_config: Dict[str, Any],
        policy_queue: mp.Queue,
        experience_queue: mp.Queue,
        stop_event: mp.Event,
        safety_bonus_coeff: float,
        worker_steps_per_iteration: int = 32,
        env_name: str = "highway-fast-v0",
    ):
        self.worker_id = worker_id
        self.highway_config = highway_config
        self.policy_queue = policy_queue
        self.experience_queue = experience_queue
        self.stop_event = stop_event
        self.safety_bonus_coeff = safety_bonus_coeff
        self.worker_steps_per_iteration = worker_steps_per_iteration

        # Worker-specific seed (since FLAGS not available in worker process)
        # Use worker_id for seed differentiation
        worker_seed = 42 + worker_id * 1000
        self.env = create_highway_env(
            highway_config, env_name=env_name, seed=worker_seed
        )

        # Initialize policy state
        self.current_policy_params = None
        self.policy_apply_fns = None

        # Worker statistics
        self.episodes_completed = 0
        self.total_steps = 0
        self.experiences_sent = 0

    def update_policy(
        self, policy_params: Dict[str, Any], apply_fns: Dict[str, Any]
    ) -> None:
        """Update worker's policy parameters."""
        self.current_policy_params = policy_params
        self.policy_apply_fns = apply_fns

    def sample_action(
        self, observation: np.ndarray, output_range: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Sample action using current policy."""
        if self.current_policy_params is None:
            # Use random actions if no policy available
            return self.env.action_space.sample()

        try:
            # Use JAX functions to sample action
            key, rng = jax.random.split(self.current_policy_params["rng"])
            dist = self.policy_apply_fns["actor"](
                {"params": self.current_policy_params["actor_params"]}, observation
            )

            # Apply limits if available
            if "limits_params" in self.current_policy_params:
                dist = self.policy_apply_fns["limits"](
                    {"params": self.current_policy_params["limits_params"]},
                    dist,
                    output_range=output_range,
                )

            action = dist.sample(seed=key)

            # Update RNG for next sampling
            self.current_policy_params["rng"] = rng

            return np.asarray(action)

        except Exception as e:
            # Debug disabled in worker processes to avoid flag access
            return self.env.action_space.sample()

    def collect_experiences(
        self, num_steps: int, output_range: Tuple[np.ndarray, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Collect experiences from environment interaction."""
        experiences = []

        for _ in range(num_steps):
            if self.stop_event.is_set():
                break

            # Get current observation
            if not hasattr(self, "current_obs"):
                self.current_obs, _ = self.env.reset()

            # Sample action
            action = self.sample_action(self.current_obs, output_range)
            action = np.clip(
                action, self.env.action_space.low, self.env.action_space.high
            )

            # Take environment step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Calculate training reward
            training_reward, reward_components = calculate_training_reward(
                self.env,
                reward,
                info,
                self.safety_bonus_coeff,
                reward_speed_range=self.highway_config["reward_speed_range"],
                next_obs=next_obs,
            )

            # Create experience
            mask = 0.0 if (done or truncated) else 1.0
            experience = {
                "observations": self.current_obs,
                "actions": action,
                "rewards": training_reward,
                "masks": mask,
                "dones": done,
                "next_observations": next_obs,
                "safety": -reward_components["safety_reward"],
            }
            experiences.append(experience)

            self.current_obs = next_obs
            self.total_steps += 1

            # Reset environment if episode ended
            if done or truncated:
                self.current_obs, _ = self.env.reset()
                self.episodes_completed += 1

        return experiences

    def check_for_policy_update(self) -> bool:
        """Check for policy updates (non-blocking)."""
        try:
            while not self.policy_queue.empty():
                policy_data = self.policy_queue.get_nowait()
                policy_params = deserialize_policy_params(policy_data["params"])
                self.update_policy(policy_params, policy_data["apply_fns"])
                return True
        except queue.Empty:
            pass
        return False

    def run(self):
        """Main worker loop."""
        # Debug disabled in worker processes to avoid flag access
        pass

        # Create action range (will be updated by policy updates)
        action_min = self.env.action_space.low
        action_max = self.env.action_space.high
        output_range = (action_min, action_max)

        try:
            while not self.stop_event.is_set():
                # Check for policy updates
                self.check_for_policy_update()

                # Collect experiences
                experiences = self.collect_experiences(
                    self.worker_steps_per_iteration, output_range
                )

                if experiences:
                    # Send experiences to learner
                    try:
                        self.experience_queue.put(
                            {
                                "worker_id": self.worker_id,
                                "experiences": experiences,
                                "worker_stats": {
                                    "episodes_completed": self.episodes_completed,
                                    "total_steps": self.total_steps,
                                },
                            },
                            timeout=1.0,
                        )
                        self.experiences_sent += len(experiences)
                    except queue.Full:
                        # Debug disabled in worker processes to avoid flag access
                        pass

        except Exception as e:
            print(f"Worker {self.worker_id}: Error in run loop: {e}")
        finally:
            # Debug disabled in worker processes to avoid flag access
            pass


def worker_process(
    worker_id: int,
    highway_config: Dict[str, Any],
    policy_queue: mp.Queue,
    experience_queue: mp.Queue,
    stop_event: mp.Event,
    safety_bonus_coeff: float,
    worker_steps_per_iteration: int = 32,
    env_name: str = "highway-fast-v0",
):
    """Worker process entry point."""
    try:
        worker = EnvironmentWorker(
            worker_id,
            highway_config,
            policy_queue,
            experience_queue,
            stop_event,
            safety_bonus_coeff,
            worker_steps_per_iteration,
            env_name,
        )
        worker.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker {worker_id}: Fatal error: {e}")


class DistributedLearner:
    """Centralized learning process handling all network updates."""

    def __init__(
        self,
        agent,
        replay_buffer: DistributedReplayBuffer,
        policy_queues: List[mp.Queue],
        experience_queue: mp.Queue,
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.policy_queues = policy_queues
        self.experience_queue = experience_queue

        # Learning statistics
        self.total_updates = 0
        self.experiences_collected = 0
        self.last_policy_broadcast = time.time()
        self.last_log_time = time.time()

        # Policy apply functions for workers
        self.apply_fns = {
            "actor": self.agent.actor.apply_fn,
        }
        if hasattr(self.agent, "limits"):
            self.apply_fns["limits"] = self.agent.limits.apply_fn
            
        # Asynchronous control
        self.stop_collection = threading.Event()
        self.collection_thread = None
        self.replay_buffer_iterator = None

    def _background_experience_collection(self) -> None:
        """Background thread for continuous experience collection."""
        while not self.stop_collection.is_set():
            try:
                # Block for a short time to get experiences
                worker_data = self.experience_queue.get(timeout=0.1)
                experiences = worker_data["experiences"]

                # Insert experiences into replay buffer
                self.replay_buffer.insert_batch(experiences)
                self.experiences_collected += len(experiences)

                if FLAGS.debug_distributed:
                    worker_stats = worker_data["worker_stats"]
                    print(
                        f"Collected {len(experiences)} experiences from worker {worker_data['worker_id']}"
                    )

            except queue.Empty:
                continue
            except Exception as e:
                if FLAGS.debug_distributed:
                    print(f"Error in background collection: {e}")
                continue
                
    def start_experience_collection(self) -> None:
        """Start background experience collection thread."""
        if self.collection_thread is None:
            self.collection_thread = threading.Thread(
                target=self._background_experience_collection,
                daemon=True
            )
            self.collection_thread.start()
            print("Started background experience collection thread")
            
    def stop_experience_collection(self) -> None:
        """Stop background experience collection thread."""
        self.stop_collection.set()
        if self.collection_thread is not None:
            self.collection_thread.join(timeout=2.0)
            print("Stopped background experience collection thread")

    def update_agent(self) -> Tuple[Any, Dict[str, float]]:
        """Perform agent updates using iterator for continuous sampling."""
        # Initialize iterator if not done yet
        if self.replay_buffer_iterator is None:
            if self.replay_buffer.size < FLAGS.batch_size * FLAGS.utd_ratio:
                return self.agent, {}
            
            # Create JAX-optimized iterator
            sample_args = {
                "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
            }
            self.replay_buffer_iterator = self.replay_buffer.get_iterator(
                queue_size=2, sample_args=sample_args
            )
            print("Initialized replay buffer iterator for continuous sampling")

        try:
            # Get batch from iterator (this is JAX-optimized and fast)
            batch = next(self.replay_buffer_iterator)
        except (StopIteration, RuntimeError):
            # Buffer may be empty or iterator exhausted, skip this update
            return self.agent, {}

        # Create output range
        action_min = np.array([-1.0, -1.0])  # Default action space
        action_max = np.array([1.0, 1.0])
        output_range = (action_min, action_max)
        
        batch_size = FLAGS.batch_size * FLAGS.utd_ratio
        mini_batch_output_range = (
            jnp.tile(output_range[0], (batch_size, 1)),
            jnp.tile(output_range[1], (batch_size, 1)),
        )

        # Update agent
        self.agent, update_info = self.agent.update(
            batch, utd_ratio=FLAGS.utd_ratio, output_range=mini_batch_output_range
        )

        self.total_updates += 1
        return self.agent, update_info

    def broadcast_policy_parameters(self) -> None:
        """Broadcast updated policy parameters to all workers."""
        try:
            # Serialize policy parameters
            serialized_params = serialize_policy_params(self.agent)
            policy_data = {
                "params": serialized_params,
                "apply_fns": self.apply_fns,
                "update_count": self.total_updates,
            }

            # Send to all workers (non-blocking)
            for policy_queue in self.policy_queues:
                try:
                    # Clear old policy updates and add new one
                    while not policy_queue.empty():
                        policy_queue.get_nowait()
                    policy_queue.put_nowait(policy_data)
                except queue.Full:
                    if FLAGS.debug_distributed:
                        print("Policy queue full, skipping update")

            self.last_policy_broadcast = time.time()

        except Exception as e:
            if FLAGS.debug_distributed:
                print(f"Error broadcasting policy: {e}")

    def should_broadcast_policy(self) -> bool:
        """Check if it's time to broadcast policy updates."""
        time_since_last = time.time() - self.last_policy_broadcast
        return time_since_last >= (1.0 / FLAGS.policy_update_frequency)

    def learning_loop(self, max_updates: int, start_training: int) -> None:
        """Asynchronous learning loop with continuous updates."""
        print("Starting asynchronous distributed learning loop")
        print(f"Target updates: {max_updates}, Start training at: {start_training} buffer size")
        
        # Start background experience collection
        self.start_experience_collection()
        
        # Wait for initial experiences
        print("Waiting for initial experiences...")
        while self.replay_buffer.size < start_training:
            time.sleep(0.1)
            if self.replay_buffer.size > 0 and self.replay_buffer.size % 1000 == 0:
                print(f"Buffer size: {self.replay_buffer.size}/{start_training}")
        
        print(f"Starting training with {self.replay_buffer.size} experiences")
        
        # Progress tracking
        start_time = time.time()
        last_update_count = 0
        
        pbar = tqdm.tqdm(
            total=max_updates,
            smoothing=0.1, 
            disable=not FLAGS.tqdm,
            dynamic_ncols=True,
            desc="Training Updates"
        )
        
        try:
            # Continuous training loop
            while self.total_updates < max_updates:
                # Perform agent update
                self.agent, update_info = self.update_agent()
                
                # Update progress bar
                updates_since_last = self.total_updates - last_update_count
                if updates_since_last > 0:
                    pbar.update(updates_since_last)
                    last_update_count = self.total_updates
                    
                # Save checkpoints periodically
                if self.total_updates > 0 and self.total_updates % FLAGS.save_checkpoint_interval == 0:
                    effective_steps = self.experiences_collected
                    self.save_checkpoint(effective_steps)
                
                # Log training metrics periodically
                if self.total_updates > 0 and self.total_updates % FLAGS.log_interval == 0:
                    elapsed_time = time.time() - self.last_log_time
                    updates_per_sec = FLAGS.log_interval / elapsed_time if elapsed_time > 0 else 0
                    
                    # Calculate effective step count based on experiences collected
                    effective_steps = self.experiences_collected
                    
                    wandb.log(
                        {f"training/{k}": v for k, v in update_info.items()},
                        step=effective_steps,
                    )
                    
                    wandb.log(
                        {
                            "distributed/experiences_collected": self.experiences_collected,
                            "distributed/total_updates": self.total_updates,
                            "distributed/replay_buffer_size": self.replay_buffer.size,
                            "distributed/updates_per_sec": updates_per_sec,
                            "distributed/effective_steps": effective_steps,
                        },
                        step=effective_steps,
                    )
                    
                    # Update progress bar description
                    pbar.set_description(
                        f"Training Updates (Buffer: {self.replay_buffer.size}, "
                        f"Exp: {self.experiences_collected}, "
                        f"UPS: {updates_per_sec:.1f})"
                    )
                    
                    self.last_log_time = time.time()
                
                # Broadcast policy updates periodically  
                if self.should_broadcast_policy():
                    self.broadcast_policy_parameters()
                    
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Error in learning loop: {e}")
            raise
        finally:
            # Stop background collection
            self.stop_experience_collection()
            pbar.close()
            
        elapsed_total = time.time() - start_time
        print(f"Distributed learning completed: {self.total_updates} updates in {elapsed_total:.1f}s")
        print(f"Final stats: {self.experiences_collected} experiences, buffer size: {self.replay_buffer.size}")
        
    def save_checkpoint(self, step: int) -> None:
        """Save training checkpoint."""
        try:
            # Handle case where wandb.run.name might be None (offline mode)
            run_name = (
                wandb.run.name
                if wandb.run.name is not None
                else f"highway_distributed_run_{FLAGS.seed}"
            )
            policy_folder = os.path.abspath(
                os.path.join(FLAGS.checkpoint_dir, run_name)
            )
            os.makedirs(policy_folder, exist_ok=True)

            # Convert config to regular dict to avoid serialization issues
            config_dict = dict(FLAGS.config)

            # Convert flags to regular dict
            flags_dict = {}
            for key, value in FLAGS.flag_values_dict().items():
                if key != "config":  # Skip config to avoid circular reference
                    try:
                        if hasattr(value, "to_dict"):
                            flags_dict[key] = value.to_dict()
                        elif isinstance(value, (str, int, float, bool)) or value is None:
                            flags_dict[key] = value
                        else:
                            flags_dict[key] = str(value)
                    except Exception as e:
                        print(f"Warning: Could not serialize flag {key}: {e}")
                        flags_dict[key] = str(value)

            param_dict = {
                "actor": self.agent.actor,
                "critic": self.agent.critic,
                "target_critic_params": self.agent.target_critic,
                "temp": self.agent.temp,
                "rng": self.agent.rng,
                "config": config_dict,
                "training_flags": flags_dict,
            }

            if hasattr(self.agent, "limits"):
                param_dict["limits"] = self.agent.limits
            if hasattr(self.agent, "q_entropy_lagrange"):
                param_dict["q_entropy_lagrange"] = self.agent.q_entropy_lagrange

            # Save main model checkpoint
            checkpoints.save_checkpoint(
                policy_folder, param_dict, step=step, keep=FLAGS.keep_checkpoints
            )

            # Save config separately using pickle
            import pickle
            config_file = os.path.join(policy_folder, f"config_{step}.pkl")
            with open(config_file, "wb") as f:
                pickle.dump(
                    {
                        "config": config_dict,
                        "training_flags": flags_dict,
                    },
                    f,
                )
            print(f"Saved checkpoint at update {step} to {policy_folder}")
        except Exception as e:
            print(f"Cannot save checkpoint: {e}")


def run_trajectory(
    agent,
    env,
    max_steps=1000,
    video: bool = False,
    output_range: Tuple[float, float] = None,
    safety_bonus_coeff=0.01,
):
    """Evaluation trajectory runner (unchanged from original)."""
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
    """Evaluation function (unchanged from original)."""
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
    """Action schedule function (unchanged from original)."""
    return min(1, 3 * i / FLAGS.max_steps - 0.5)


def main(_):
    print("Starting Distributed SAC Training")
    print(
        f"Configuration: {FLAGS.num_workers} workers, {FLAGS.worker_steps_per_iteration} steps/iter"
    )

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
        "collision_reward": -10.0,
        # "collision_reward": 0.0,  # Debug: no collision penalty
        "right_lane_reward": 0.01,
        "high_speed_reward": 0.7,
        "lane_change_reward": 0.0,
        "reward_speed_range": [10, 40],
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "normalize_reward": False,
        "offroad_terminal": True,
    }

    # Create evaluation environment
    eval_env = create_highway_env(
        highway_config, env_name=FLAGS.env_name, render_mode="rgb_array"
    )
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)

    # Create agent
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    safety_bonus_coeff = kwargs.pop("safety_penalty", 0.0)
    kwargs.pop("group_name", None)
    kwargs.pop("max_offroad_steps", None)

    # Create a temporary environment to get spaces
    temp_env = create_highway_env(highway_config, env_name=FLAGS.env_name)
    agent = globals()[model_cls].create(
        FLAGS.seed, temp_env.observation_space, temp_env.action_space, **kwargs
    )
    temp_env.close()

    # Initialize W&B
    wandb_group_name = f"{FLAGS.config.group_name}"
    wandb.init(
        project=FLAGS.wandb_project,
        notes=f"Distributed SAC training with {FLAGS.num_workers} workers",
        group=(
            f"{wandb_group_name}-highway-distributed"
            if FLAGS.group_name_suffix is None
            else f"{wandb_group_name}-{FLAGS.group_name_suffix}-highway-distributed"
        ),
    )

    config_for_wandb = {
        **FLAGS.flag_values_dict(),
        "config": dict(FLAGS.config),
    }
    wandb.config.update(config_for_wandb)

    # Create distributed replay buffer
    replay_buffer = DistributedReplayBuffer(
        temp_env.observation_space,
        temp_env.action_space,
        FLAGS.replay_buffer_size,
        extra_fields=["safety"],
    )
    replay_buffer.seed(FLAGS.seed)

    # Create communication queues
    policy_queues = [mp.Queue(maxsize=2) for _ in range(FLAGS.num_workers)]
    experience_queue = mp.Queue(maxsize=FLAGS.max_queue_size)
    stop_event = mp.Event()

    # Start worker processes
    worker_processes = []
    for worker_id in range(FLAGS.num_workers):
        process = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                highway_config,
                policy_queues[worker_id],
                experience_queue,
                stop_event,
                safety_bonus_coeff,
                FLAGS.worker_steps_per_iteration,
                FLAGS.env_name,
            ),
        )
        process.start()
        worker_processes.append(process)

    print(f"Started {len(worker_processes)} worker processes")

    # Create distributed learner
    learner = DistributedLearner(agent, replay_buffer, policy_queues, experience_queue)

    # Setup signal handling for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down distributed training...")
        stop_event.set()
        for process in worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run distributed learning (using max_steps as target updates)
        target_updates = FLAGS.max_steps // FLAGS.learner_update_ratio  # Convert steps to updates
        learner.learning_loop(target_updates, FLAGS.start_training)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        # Cleanup
        print("Cleaning up distributed training...")
        stop_event.set()

        # Wait for workers to finish
        for process in worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                print(f"Force terminating worker process {process.pid}")
                process.terminate()

        print("Distributed training completed")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    app.run(main)
