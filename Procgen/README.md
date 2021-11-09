# RADIAL-RL: Procgen PPO

This repo is an implementation of our ProcGen experiments for our Neurips 2021 submission: Robust Deep Reinforcement Learning through Adversarial Loss.

This repository is based on [train-procgen-pfrl](https://github.com/lerrytang/train-procgen-pfrl).

[Procgen](https://openai.com/blog/procgen-benchmark/) is a suite of
16 procedurally-generated environments that serves as a benchmark to measure how
quickly a reinforcement learning agent learns generalizable skills.

## Requirements
To run our code you need to have Python 3 (>=3.7) and pip installed on your systems. Additionally we require PyTorch>=1.4, which should be installed using instructions from https://pytorch.org/get-started/locally/.

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-trained Models

Pre-trained models are available at https://www.dropbox.com/sh/tgjfje9alozwtf8/AAAPjBVO8tVXFl_ktpVipj1ma?dl=0, download and save to `.\trained_models`.

## Training

To train a standard PPO model on FruitBot with 200 training levels like the one used on our paper, run the following command:

```train PPO
python train_procgen
```


## Robust training

To train a robust DQN model on CoinRun like the one we reported against 1/255 perturbations, use the following:

```RADIAL-PPO
python train_procgen.py --env-name=coinrun --epsilon-end 3.92e-3
```

By default models are trained and evaluated on GPU 0, to train using CPU pass `--gpu -1`

## Evaluation

To evaluate our robustly trained Coinrun model with deterministic policy, use the following command:

```
python eval_procgen.py --env-name=coinrun --model-file=trained_models/CoinRun_radial_ppo.pt --standard --pgd --deterministic
```

You can evaluate stochastic policy performance by dropping the `--deterministic` flag. If you only want to evaluate standard performance(much faster), drop the `--pgd` flag.

### Training commands for models reported

| Game       | Model      |                                                                                          Command                                                                                          |
|------------|------------|:------------------------------------------------------------------------------------------------------------------------:|
|    FruitBot    | PPO |               python train_procgen.py --env-name=fruitbot --exp-name ppo_trained |
|            | RADIAL-PPO |             python train_procgen.py --env-name=fruitbot --exp-name radial_ppo_trained --epsilon-end 3.92e-3  |
|   CoinRun  | PPO |    python train_procgen.py --env-name=coinrun --exp-name ppo_trained   |
|   | RADIAL-PPO |  python train_procgen.py --env-name=coinrun --exp-name radial_ppo_trained --epsilon-end 3.92e-3  |
|  Jumper | PPO |       python train_procgen.py --env-name=jumper --exp-name ppo_trained |
| | RADIAL-PPO | python train_procgen.py --env-name=jumper --exp-name radial_ppo_trained --epsilon-end 3.92e-3 |

