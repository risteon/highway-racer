import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "highway-env_distributional"
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
    # This placeholder is filled in to -action_dim/2
    # config.target_entropy = config_dict.placeholder(float)
    # increase this value to encourage exploration, e.g. discover overtaking
    config.target_entropy = -2.0

    config.num_atoms = 151
    config.q_min = -40.0
    config.q_max = 80.0

    config.cvar_risk = 0.5

    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.limits_weight_decay = 1e-3

    config.backup_entropy = False
    config.independent_ensemble = True

    config.q_entropy_target_diff = 0.5
    config.q_entropy_lagrange_init = 1e-3
    config.q_entropy_lagrange_lr = 1e-4

    # PointNet-specific parameters
    config.use_pointnet = True
    config.pointnet_hidden_dims = (128,)
    config.pointnet_reduce_fn = "max"
    config.pointnet_use_layer_norm = True
    config.pointnet_dropout_rate = None

    return config
