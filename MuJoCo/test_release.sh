#!/bin/bash
conda deactivate
source env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wang/.mujoco/mjpro150/bin

python src/test.py --config-path src/config_halfcheetah_radial_ppo.json --load-model radial_ppo_pretrained_model/half_cheetah.model --deterministic
python src/test.py --config-path src/config_walker_radial_ppo.json --load-model radial_ppo_pretrained_model/walker.model --deterministic
python src/test.py --config-path src/config_hopper_radial_ppo.json --load-model radial_ppo_pretrained_model/hopper.model --deterministic

python src/test.py --config-path src/config_halfcheetah_radial_ppo.json --load-model radial_ppo_pretrained_model/half_cheetah.model --deterministic --attack-eps=0.075 --attack-method action
python src/test.py --config-path src/config_walker_radial_ppo.json --load-model radial_ppo_pretrained_model/walker.model --deterministic  --attack-eps=0.075 --attack-method action
python src/test.py --config-path src/config_hopper_radial_ppo.json --load-model radial_ppo_pretrained_model/hopper.model --deterministic --attack-eps=0.075 --attack-method action
