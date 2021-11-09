import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["robust_ppo"],
    "out_dir": ["robust_ppo_convex_walker_best/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [4e-4],
    "val_lr": [3e-4],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [10],
    "robust_ppo_eps": [0.05],
    "robust_ppo_reg": [0.1] * 15,
    "train_steps": [976],
    "robust_ppo_eps_scheduler_opts": ["start=1,length=732"],
    "robust_ppo_beta": [1.0],
    "robust_ppo_beta_scheduler_opts": ["same"], # Using the same scheduler as eps scheduler
    "robust_ppo_detach_stdev": [False], 
    "robust_ppo_method": ["convex-relax"],
}

generate_configs(BASE_CONFIG, PARAMS)
