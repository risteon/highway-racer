import threading
import queue


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
        next_observations, rewards, terminations, truncations, infos = self.envs[
            index
        ].step(actions)
        # Handle episode terminations and create masks
        # IMPORTANT: For Q-value bootstrapping, only use terminations (not truncations)
        # - Terminated episodes: shouldn't bootstrap (mask=0.0) - true episode end
        # - Truncated episodes: should bootstrap (mask=1.0) - episode continues but time limit hit
        dones = (
            terminations | truncations
        )  # Combined done flag for episode boundary detection
        masks = 1.0 - terminations.astype(
            float
        )  # Only terminations prevent bootstrapping

        if np.any(dones):
            # Handle `final_observation` in case of a truncation or termination
            real_next_observations = next_observations.copy()
            for idx in np.where(dones)[0]:
                real_next_observations[idx] = infos["final_obs"][idx]

            # add penalty for offroad terminations
            offroad_terminations = (
                infos["final_info"]["rewards"]["on_road_reward"] == 0.0
            )
            OFFROAD_PENALTY = -2.0
            rewards += offroad_terminations * OFFROAD_PENALTY

            episode_infos = infos["final_info"]["episode"]
            r = (episode_infos["r"] + offroad_terminations * OFFROAD_PENALTY)[
                dones
            ].mean()
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

            self.offroad_termination_counter += np.sum(offroad_terminations[dones])

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
