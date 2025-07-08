import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "highway_pointnet_distributional"

    config.model_cls = "DistributionalSACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    # Standard MLP dimensions (final layers after PointNet)
    config.critic_hidden_dims = (256, 256)
    config.actor_hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 2

    config.tau = 0.005
    config.init_temperature = 1.0
    # config.target_entropy = config_dict.placeholder(float)
    # config.target_entropy = -0.5
    # might use:
    config.target_entropy = -2.0

    # Adjusted distributional parameters for highway environment
    config.num_atoms = 151
    config.q_min = -40.0  # Adjusted for highway rewards (crashes, penalties)
    # config.q_max = 60.0  # Adjusted for highway rewards (goal completion)
    # config.q_max = 90.0  # Increased, better
    # config.q_max = 120.0  # Increased.
    config.q_max = 80.0  # Lower speed return.
    # config.cvar_risk = 0.0
    # config.cvar_risk = 0.25
    config.cvar_risk = 0.5
    # config.cvar_risk = 0.6
    # config.cvar_risk = 0.8
    # config.cvar_risk = 0.9

    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.limits_weight_decay = 1e-3

    config.backup_entropy = False
    config.independent_ensemble = True  # new

    # config.q_entropy_target_diff = -0.01  # new, prev 0.5 default
    config.q_entropy_target_diff = 0.5  # new, try default for convergence
    config.q_entropy_lagrange_init = 1e-3  # new
    config.q_entropy_lagrange_lr = 1e-4  # new. 1e-3 in other config?

    # Highway-specific safety penalty
    config.safety_penalty = 0.0  # Small safety bonus coefficient

    # PointNet-specific parameters
    config.use_pointnet = True
    config.pointnet_hidden_dims = (128,)
    config.pointnet_reduce_fn = "max"
    config.pointnet_use_layer_norm = True
    config.pointnet_dropout_rate = None

    return config
