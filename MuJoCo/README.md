# RADIAL-PPO: Robust ADversarIAl Loss PPO for robust deep reinforcement learning
This repo is an implementation of our MuJoCo experiments for our Neurips 2021 submission: Robust Deep Reinforcement Learning through Adversarial Loss.
During policy optimization, we perturb the policy network and use the worst-case value function estimation to improve the robustness of RL agent under adversarial attack.
The code is based on [SA-PPO](https://github.com/huanzhang12/SA_PPO).

## setup environment
The required packages are same as [SA-PPO](https://github.com/huanzhang12/SA_PPO).

First clone this repository and install necessary Python packages:

```bash
git submodule update --init
pip install -r requirements.txt
```

Python 3.7+ is required. Note that you need to install MuJoCo 1.5 first to use
the Gym environments.  See
[here](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#requirements)
for instructions.

## run experiment
to run the default configuration of radial-ppo training
```bash
bash train_release.sh
```


## test pre-trained models
to test the pre-trained radial-ppo models
```bash
bash test_release.sh
```

