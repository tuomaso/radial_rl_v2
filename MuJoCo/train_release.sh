#!/bin/bash

python src/run.py --config-path src/config_walker_radial_ppo.json
python src/run.py --config-path src/config_hopper_radial_ppo.json
python src/run.py --config-path src/config_halfcheetah_radial_ppo.json
