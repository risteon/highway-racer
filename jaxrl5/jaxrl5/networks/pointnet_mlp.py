from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl5.networks import default_init


class PointNetMLP(nn.Module):
    """
    PointNet-style MLP for permutation-invariant processing of multi-vehicle observations.

    Processes observations of shape (N, 6) where:
    - First vehicle (index 0) is ego vehicle
    - Remaining vehicles (indices 1:N) are other vehicles
    - Applies PointNet-style processing to other vehicles for permutation invariance

    Architecture:
    1. Split ego features from other vehicles
    2. Apply per-vehicle MLP to other vehicles using vmap
    3. Aggregate other vehicle features using reduce function
    4. Concatenate ego features with aggregated others
    5. Apply final MLP
    """

    # PointNet-specific parameters for other vehicle processing
    pointnet_hidden_dims: Sequence[int] = (128, 128)
    pointnet_activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    pointnet_use_layer_norm: bool = False
    pointnet_dropout_rate: Optional[float] = None

    # Final MLP parameters (after concatenation)
    final_hidden_dims: Sequence[int] = (256, 256)
    final_activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    final_activate_final: bool = False
    final_use_layer_norm: bool = False
    final_scale_final: Optional[float] = None
    final_dropout_rate: Optional[float] = None

    # Configuration
    ego_features: int = 6
    vehicle_features: int = 6
    reduce_fn: str = "max"  # "max", "mean", "sum"

    # Critic-specific configuration
    is_critic: bool = (
        False  # Whether this is used in critic (with action concatenation)
    )
    action_dim: int = 2  # Action dimension for critic mode

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass of PointNetMLP.

        Args:
            x: Input data, depends on mode:
               - Actor mode: observation of shape (N, 6), (N*6,), (B, N, 6), or (B, N*6)
               - Critic mode: [observations, actions] of shape (N*6 + action_dim,) or (B, N*6 + action_dim)
            training: Whether in training mode (for dropout)

        Returns:
            Output features after PointNet processing and final MLP
        """
        if self.is_critic:
            # Critic mode: split observations and actions
            obs, actions = self._split_obs_actions(x)
            obs = self._preprocess_input(obs)

            # Handle batched vs single sample
            if len(obs.shape) == 3:
                # Batched: (B, N, 6)
                batch_size = obs.shape[0]
                ego_features = obs[:, 0, :]  # (B, 6)
                other_vehicles = obs[:, 1:, :]  # (B, N-1, 6)

                # Process other vehicles with PointNet (batched)
                aggregated_others = self._pointnet_process_batched(
                    other_vehicles, training
                )

                # Combine ego, aggregated others, and actions
                combined_features = jnp.concatenate(
                    [ego_features, aggregated_others, actions], axis=-1
                )
            else:
                # Single sample: (N, 6)
                ego_features = obs[0]  # (6,)
                other_vehicles = obs[1:]  # (N-1, 6)

                # Process other vehicles with PointNet
                aggregated_others = self._pointnet_process(other_vehicles, training)

                # Combine ego, aggregated others, and actions
                combined_features = jnp.concatenate(
                    [ego_features, aggregated_others, actions]
                )
        else:
            # Actor mode: standard PointNet processing
            obs = self._preprocess_input(x)

            # Handle batched vs single sample
            if len(obs.shape) == 3:
                # Batched: (B, N, 6)
                batch_size = obs.shape[0]
                ego_features = obs[:, 0, :]  # (B, 6)
                other_vehicles = obs[:, 1:, :]  # (B, N-1, 6)

                # Process other vehicles with PointNet (batched)
                aggregated_others = self._pointnet_process_batched(
                    other_vehicles, training
                )

                # Combine ego and aggregated others
                combined_features = jnp.concatenate(
                    [ego_features, aggregated_others], axis=-1
                )
            else:
                # Single sample: (N, 6)
                ego_features = obs[0]  # (6,)
                other_vehicles = obs[1:]  # (N-1, 6)

                # Process other vehicles with PointNet
                aggregated_others = self._pointnet_process(other_vehicles, training)

                # Combine ego and aggregated others
                combined_features = jnp.concatenate([ego_features, aggregated_others])

        # Apply final MLP
        return self._final_mlp(combined_features, training)

    def _pointnet_process(
        self, other_vehicles: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        """
        Apply PointNet-style processing to other vehicles.

        Args:
            other_vehicles: Other vehicle features of shape (N-1, 6)
            training: Whether in training mode

        Returns:
            Aggregated features of shape (pointnet_hidden_dims[-1],)
        """
        # Handle case with no other vehicles
        if other_vehicles.shape[0] == 0:
            return jnp.zeros(self.pointnet_hidden_dims[-1])

        # Apply per-vehicle MLP using vmap for parallel processing
        def process_single_vehicle(vehicle_features):
            return self._pointnet_mlp(vehicle_features, training)

        # Process all vehicles in parallel
        vehicle_embeddings = jax.vmap(process_single_vehicle)(other_vehicles)
        # Shape: (N-1, pointnet_hidden_dims[-1])

        # Aggregate using specified reduction function
        if self.reduce_fn == "max":
            aggregated = jnp.max(vehicle_embeddings, axis=0)
        elif self.reduce_fn == "mean":
            aggregated = jnp.mean(vehicle_embeddings, axis=0)
        elif self.reduce_fn == "sum":
            aggregated = jnp.sum(vehicle_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown reduce_fn: {self.reduce_fn}")

        return aggregated

    def _pointnet_process_batched(
        self, other_vehicles: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        """
        Apply PointNet-style processing to batched other vehicles.

        Args:
            other_vehicles: Other vehicle features of shape (B, N-1, 6)
            training: Whether in training mode

        Returns:
            Aggregated features of shape (B, pointnet_hidden_dims[-1])
        """
        batch_size, n_vehicles, _ = other_vehicles.shape

        # Handle case with no other vehicles
        if n_vehicles == 0:
            return jnp.zeros((batch_size, self.pointnet_hidden_dims[-1]))

        # Apply per-vehicle MLP using vmap
        def process_single_vehicle(vehicle_features):
            return self._pointnet_mlp(vehicle_features, training)

        # Reshape to apply vmap: (B, N-1, 6) -> (B*N-1, 6)
        batch_size, n_vehicles, feature_dim = other_vehicles.shape
        flattened_vehicles = other_vehicles.reshape(
            batch_size * n_vehicles, feature_dim
        )

        # Apply MLP to all vehicles: (B*N-1, 6) -> (B*N-1, hidden_dim)
        # vehicle_vmap = jax.vmap(process_single_vehicle)
        # flattened_embeddings = vehicle_vmap(flattened_vehicles)
        flattened_embeddings = process_single_vehicle(flattened_vehicles)

        # Reshape back: (B*N-1, hidden_dim) -> (B, N-1, hidden_dim)
        vehicle_embeddings = flattened_embeddings.reshape(
            batch_size, n_vehicles, self.pointnet_hidden_dims[-1]
        )

        # Aggregate using specified reduction function over vehicle dimension
        if self.reduce_fn == "max":
            aggregated = jnp.max(vehicle_embeddings, axis=1)  # (B, hidden_dim)
        elif self.reduce_fn == "mean":
            aggregated = jnp.mean(vehicle_embeddings, axis=1)  # (B, hidden_dim)
        elif self.reduce_fn == "sum":
            aggregated = jnp.sum(vehicle_embeddings, axis=1)  # (B, hidden_dim)
        else:
            raise ValueError(f"Unknown reduce_fn: {self.reduce_fn}")

        return aggregated

    def _pointnet_mlp(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Per-vehicle MLP processing for PointNet.

        Args:
            x: Single vehicle features of shape (6,)
            training: Whether in training mode

        Returns:
            Vehicle embedding of shape (pointnet_hidden_dims[-1],)
        """
        for i, size in enumerate(self.pointnet_hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            # Always apply activation for PointNet layers
            if (
                self.pointnet_dropout_rate is not None
                and self.pointnet_dropout_rate > 0
            ):
                x = nn.Dropout(rate=self.pointnet_dropout_rate)(
                    x, deterministic=not training
                )
            if self.pointnet_use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.pointnet_activations(x)

        return x

    def _final_mlp(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Final MLP processing after concatenating ego and aggregated other features.

        This follows the same pattern as the existing MLP implementation.

        Args:
            x: Combined features of shape (ego_features + pointnet_hidden_dims[-1],)
            training: Whether in training mode

        Returns:
            Final output features
        """
        for i, size in enumerate(self.final_hidden_dims):
            if (
                i + 1 == len(self.final_hidden_dims)
                and self.final_scale_final is not None
            ):
                x = nn.Dense(size, kernel_init=default_init(self.final_scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.final_hidden_dims) or self.final_activate_final:
                if self.final_dropout_rate is not None and self.final_dropout_rate > 0:
                    x = nn.Dropout(rate=self.final_dropout_rate)(
                        x, deterministic=not training
                    )
                if self.final_use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.final_activations(x)

        return x

    def _preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Handle both flattened and unflattened inputs, with or without batch dimension.

        Supports:
        - Single sample: (N*6,) -> (N, 6) or (N, 6) -> (N, 6)
        - Batched: (B, N*6) -> (B, N, 6) or (B, N, 6) -> (B, N, 6)

        Args:
            x: Input observation, with various possible shapes

        Returns:
            Reshaped observation of shape (N, 6) or (B, N, 6)
        """
        if len(x.shape) == 1:
            # Single sample, flattened: (N*6,) -> (N, 6)
            if x.shape[0] % self.vehicle_features != 0:
                raise ValueError(
                    f"Flattened input size {x.shape[0]} not divisible by "
                    f"vehicle_features {self.vehicle_features}"
                )
            n_vehicles = x.shape[0] // self.vehicle_features
            return x.reshape(n_vehicles, self.vehicle_features)
        elif len(x.shape) == 2:
            if x.shape[1] == self.vehicle_features:
                # Single sample, already correct: (N, 6) -> (N, 6)
                return x
            elif x.shape[1] % self.vehicle_features == 0:
                # Batched, flattened: (B, N*6) -> (B, N, 6)
                batch_size = x.shape[0]
                n_vehicles = x.shape[1] // self.vehicle_features
                return x.reshape(batch_size, n_vehicles, self.vehicle_features)
            else:
                raise ValueError(
                    f"2D input last dimension {x.shape[1]} not divisible by "
                    f"vehicle_features {self.vehicle_features}"
                )
        elif len(x.shape) == 3 and x.shape[2] == self.vehicle_features:
            # Batched, already correct: (B, N, 6) -> (B, N, 6)
            return x
        else:
            raise ValueError(
                f"Unexpected input shape: {x.shape}. Expected one of: "
                f"(N*{self.vehicle_features},), (N, {self.vehicle_features}), "
                f"(B, N*{self.vehicle_features}), or (B, N, {self.vehicle_features})"
            )

    def _split_obs_actions(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Split concatenated observations and actions for critic mode.

        Args:
            x: Concatenated [observations, actions] of shape:
               - Single: (N*6 + action_dim,)
               - Batched: (B, N*6 + action_dim)

        Returns:
            obs: Observations of shape (N*6,) or (B, N*6)
            actions: Actions of shape (action_dim,) or (B, action_dim)
        """
        if len(x.shape) == 1:
            # Single sample
            if x.shape[0] < self.action_dim:
                raise ValueError(
                    f"Input too small for critic mode. Expected at least {self.action_dim} "
                    f"elements for actions, got {x.shape[0]}"
                )

            # Split observations and actions
            obs_size = x.shape[0] - self.action_dim
            observations = x[:obs_size]  # First part: observations
            actions = x[obs_size:]  # Last part: actions

        elif len(x.shape) == 2:
            # Batched
            if x.shape[1] < self.action_dim:
                raise ValueError(
                    f"Input too small for critic mode. Expected at least {self.action_dim} "
                    f"elements for actions, got {x.shape[1]}"
                )

            # Split observations and actions
            obs_size = x.shape[1] - self.action_dim
            observations = x[:, :obs_size]  # First part: observations (B, N*6)
            actions = x[:, obs_size:]  # Last part: actions (B, action_dim)

        else:
            raise ValueError(
                f"Expected 1D or 2D input for critic mode, got shape {x.shape}"
            )

        return observations, actions
