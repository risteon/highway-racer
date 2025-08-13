#! /usr/bin/env python
from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
from typing import Tuple

from flax.training import checkpoints
import numpy as np
import tqdm
import wandb
import tyro

from jax import numpy as jnp
from jaxrl5.agents import *
from jaxrl5.data import ReplayBuffer

from ..trainer.env import EnvPair, make_env
from ..trainer.async_env_stepper import AsyncEnvStepper


@dataclass
class Args:
    env_name: str = field(
        default="highway-fast-v0", metadata={"help": "Highway environment name."}
    )
    wandb_project: str = field(
        default="highway_racer", metadata={"help": "Project for W&B"}
    )
    comment: str = field(default="", metadata={"help": "Comment for W&B"})
    save_dir: str = field(
        default="./tmp/", metadata={"help": "Tensorboard logging dir."}
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})
    eval_episodes: int = field(
        default=16, metadata={"help": "Number of episodes used for evaluation."}
    )
    log_interval: int = field(default=500, metadata={"help": "Logging interval."})
    eval_interval: int = field(default=25000, metadata={"help": "Eval interval."})
    batch_size: int = field(default=128, metadata={"help": "Mini batch size."})
    max_steps: int = field(
        default=int(2e6), metadata={"help": "Number of training steps."}
    )
    start_training: int = field(
        default=int(1e3),
        metadata={"help": "Number of training steps to start training."},
    )
    replay_buffer_size: int = field(
        default=30000, metadata={"help": "Capacity of the replay buffer."}
    )
    tqdm: bool = field(default=True, metadata={"help": "Use tqdm progress bar."})
    save_video: bool = field(
        default=False, metadata={"help": "Save videos during evaluation."}
    )
    record_video: bool = field(
        default=False, metadata={"help": "Record videos during training."}
    )
    save_buffer: bool = field(
        default=False, metadata={"help": "Save the replay buffer."}
    )
    save_buffer_interval: int = field(
        default=50000, metadata={"help": "Save buffer interval."}
    )
    save_checkpoint_interval: int = field(
        default=5000, metadata={"help": "Steps between saving checkpoints."}
    )
    checkpoint_dir: str = field(
        default="policies", metadata={"help": "Directory to save checkpoints."}
    )
    keep_checkpoints: int = field(
        default=41, metadata={"help": "Number of checkpoints to keep."}
    )
    utd_ratio: int = field(default=8, metadata={"help": "Updates per data point"})
    reset_interval: int = field(
        default=100000,
        metadata={
            "help": "Parameter reset interval, in network updates (= env steps * UTD)"
        },
    )
    reset_ensemble: bool = field(
        default=False, metadata={"help": "Reset one ensemble member at a time"}
    )
    num_envs: int = field(
        default=8, metadata={"help": "Number of parallel environments"}
    )
    config: str = field(
        default="distributional_sac_cvar.py",
        metadata={"help": "File path to the training hyperparameter configuration."},
    )


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


def main(args: Args):
    envs = EnvPair(args.env_name, args.seed, args.num_envs)
    eval_env = make_env(
        args.env_name, args.seed, idx=args.num_envs * 2, render_mode="rgb_array"
    )()

    spec = importlib.util.spec_from_file_location(
        "config", Path("racer") / "configs" / args.config
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()

    agent = globals()[config.model_cls].create(
        args.seed, envs.first.observation_space, envs.first.action_space, **config
    )

    replay_buffer = ReplayBuffer(
        envs.first.single_observation_space,
        envs.first.single_action_space,
        args.replay_buffer_size,
    )
    replay_buffer.seed(args.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": args.batch_size * args.utd_ratio,
        }
    )

    wandb.init(project=args.wandb_project, notes=args.comment, group=config.group_name)
    config_for_wandb = {
        **args.__dict__,
        "config": dict(config),
    }
    wandb.config.update(config_for_wandb)

    reset_interval = args.reset_interval
    if args.reset_ensemble:
        reset_interval = reset_interval // agent.num_qs

    action_min: np.ndarray = envs.first.single_action_space.low
    action_max: np.ndarray = envs.first.single_action_space.high

    pbar = tqdm.tqdm(
        range(1, args.max_steps + 1),
        smoothing=0.1,
        disable=not args.tqdm,
        dynamic_ncols=True,
    )

    # Create async environment stepper
    async_stepper = AsyncEnvStepper(
        envs, replay_buffer, log_interval=args.log_interval, seed=args.seed
    )
    async_stepper.start()
    observations, index = async_stepper.get_results(timeout=None)

    try:
        for i in pbar:
            output_range = (action_min, action_max)

            if i < args.start_training:
                actions = envs.first.action_space.sample()
            else:
                # Sample actions from agent using current observations
                actions, agent = agent.sample_actions(
                    observations, output_range=output_range
                )
                actions = np.array(actions)

            # Queue actions for async environment stepping
            if async_stepper.queue_actions(actions, i, index):
                # Actions queued successfully
                pass
            else:
                # Queue full, skip this step or handle differently
                print(f"Warning: Action queue full at step {i}")

            observations, index = async_stepper.get_results(timeout=None)

            if i >= args.start_training:
                batch = next(replay_buffer_iterator)

                output_range = (action_min, action_max)

                mini_batch_output_range = (
                    jnp.tile(output_range[0], (args.batch_size * args.utd_ratio, 1)),
                    jnp.tile(output_range[1], (args.batch_size * args.utd_ratio, 1)),
                )

                agent, update_info = agent.update(
                    batch,
                    utd_ratio=args.utd_ratio,
                    output_range=mini_batch_output_range,
                )

                if i % args.log_interval == 0:
                    # Handle Q-value distribution visualization
                    log_dict = {}
                    for k, v in update_info.items():
                        if k == "critic_value_hist":
                            # Handle Q-value distribution visualization
                            probs, atoms = v

                            # Convert to numpy
                            probs_np = np.array(probs).flatten()
                            atoms_np = np.array(atoms).flatten()

                            # Create table for bar chart
                            data = [
                                [float(atom), float(prob)]
                                for atom, prob in zip(atoms_np, probs_np)
                            ]
                            table = wandb.Table(
                                data=data, columns=["q_value", "probability"]
                            )

                            log_dict[f"training/q_distribution_bar"] = wandb.plot.bar(
                                table,
                                "q_value",
                                "probability",
                                title="Q-Value Distribution",
                            )
                        else:
                            log_dict[f"training/{k}"] = v

                    wandb.log(log_dict, step=i)

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
                    and i < int(args.max_steps * 0.8)
                ):
                    if args.reset_ensemble:
                        agent, ensemble_info = agent.reset_ensemble_member()
                        for k, v in ensemble_info.items():
                            wandb.log({f"reset/{k}": v}, step=i)
                    else:
                        agent = agent.reset(exclude=["critic", "target_critic"])

            if i % args.eval_interval == 0:
                evaluate(
                    agent,
                    eval_env,
                    args.eval_episodes,
                    output_range=output_range,
                )

            # Save checkpoints at specified intervals
            if i % args.save_checkpoint_interval == 0 or i == 100:
                try:
                    # Handle case where wandb.run.name might be None (offline mode)
                    run_name = (
                        wandb.run.name
                        if wandb.run.name is not None
                        else f"highway_run_{args.seed}"
                    )
                    policy_folder = os.path.abspath(
                        os.path.join(args.checkpoint_dir, run_name)
                    )
                    os.makedirs(policy_folder, exist_ok=True)

                    # Convert flags to regular dict
                    flags_dict = {}
                    for key, value in args.__dict__.items():
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
                        "config": dict(config),
                        "training_flags": flags_dict,  # Save all training flags as regular dict
                    }

                    # Config and flags prepared for saving
                    if hasattr(agent, "limits"):
                        param_dict["limits"] = agent.limits
                    if hasattr(agent, "q_entropy_lagrange"):
                        param_dict["q_entropy_lagrange"] = agent.q_entropy_lagrange

                    # Save main model checkpoint using Orbax
                    checkpoints.save_checkpoint(
                        policy_folder, param_dict, step=i, keep=args.keep_checkpoints
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

            if args.save_buffer and i % args.save_buffer_interval == 0:
                with open(os.path.join(args.save_dir, f"buffer_{i}.pkl"), "wb") as f:
                    pickle.dump(replay_buffer, f)

    finally:
        # Stop async stepper and clean up
        print("Stopping async environment stepper...")
        async_stepper.stop()

        # Process any remaining results
        try:
            while True:
                remaining_result = async_stepper.get_results(timeout=0.1)
        except:
            pass

    if args.save_buffer:
        with open(os.path.join(args.save_dir, "final_buffer.pkl"), "wb") as f:
            pickle.dump(replay_buffer, f)

    envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
