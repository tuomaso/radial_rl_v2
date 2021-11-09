import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["ppo"],
    "out_dir": ["vanilla_ppo_humanoid/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [5e-5] * 72,
    "val_lr": [1e-5],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [100],
    "train_steps": [4882],
    "robust_ppo_eps": [0.075], # used for attack
}

generate_configs(BASE_CONFIG, PARAMS)
