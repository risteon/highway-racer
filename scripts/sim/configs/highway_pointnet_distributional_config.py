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
    config.target_entropy = config_dict.placeholder(float)

    # Adjusted distributional parameters for highway environment
    config.num_atoms = 151  # Sufficient for highway reward range
    config.q_min = -10.0  # Adjusted for highway rewards (crashes, penalties)
    config.q_max = 150.0  # Adjusted for highway rewards (goal completion)
    config.cvar_risk = 0.9

    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.limits_weight_decay = 1e-3

    config.backup_entropy = False

    # Highway-specific safety penalty
    config.safety_penalty = 0.0  # Small safety bonus coefficient

    # PointNet-specific parameters
    config.use_pointnet = True
    config.pointnet_hidden_dims = (128, 128)
    config.pointnet_reduce_fn = "max"
    config.pointnet_use_layer_norm = False
    config.pointnet_dropout_rate = None

    return config