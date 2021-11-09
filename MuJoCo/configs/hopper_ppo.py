import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Hopper-v2"],
    "mode": ["ppo"],
    "out_dir": ["vanilla_ppo_hopper/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [False],
    "ppo_lr_adam": [3e-4] * 40,
    "val_lr": [2.5e-4],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [10],
    "train_steps": [976],
    "robust_ppo_eps": [0.075], # used for attack
}

generate_configs(BASE_CONFIG, PARAMS)
