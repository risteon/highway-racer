import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "highway_distributional"

    config.model_cls = "DistributionalSACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

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
    # config.safety_penalty = 0.1  # Moderate safety bonus coefficient
    # config.safety_penalty = 0.01  # Small safety bonus coefficient
    config.safety_penalty = 0.0  # Small safety bonus coefficient

    return config
