import dataclasses
from holosoma.config_values.experiment import DEFAULTS
from holosoma.utils.helpers import get_class
from holosoma.train_agent import get_tyro_env_config, training_context
from holosoma.utils.common import seeding
import torch

seeding(0)
num_envs = 16
device = "cuda"

cfg = dataclasses.replace(
    DEFAULTS["brs_v1"],
    training=dataclasses.replace(DEFAULTS["brs_v1"].training, num_envs=num_envs)
)

with training_context(cfg):
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
    obs = env.reset_all()
    actions = torch.zeros(num_envs, cfg.robot.actions_dim, device=device)
    for _ in range(100):
        obs, rew, dones, info = env.step({"actions": actions})
    print("E2E OK — obs shape:", obs["actor_obs"].shape)