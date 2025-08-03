import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "highway-env_sac"
    config.model_cls = "SACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99
    config.num_qs = 2

    config.tau = 0.005
    config.init_temperature = 1.0
    # increase this value to encourage exploration, e.g. discover overtaking
    config.target_entropy = -2.0

    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.backup_entropy = False

    # PointNet-specific parameters
    config.use_pointnet = True
    config.pointnet_hidden_dims = (128,)
    config.pointnet_reduce_fn = "max"
    config.pointnet_use_layer_norm = True
    config.pointnet_dropout_rate = None

    return config
