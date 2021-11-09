#!/bin/bash

trap "kill 0" SIGINT  # exit cleanly when pressing control+C
# Set number of threads if necessary
# export nthreads=128
source scan_attacks.sh

# Generate a random semaphore ID (don't touch it)
semaphorename=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

# Set this flag so that the sem in attack scan does not wait.
export ATTACK_FOLDER_NO_WAIT=1

export ATTACK_MODEL_NO_STOCHASTIC=1

# Scan folders. Check carefully about folder name and config file name.
scan_exp_folder config_ant_vanilla_ppo.json adv_ppo_ant/agents $semaphorename
scan_exp_folder config_hopper_vanilla_ppo.json adv_ppo_hopper/agents $semaphorename
scan_exp_folder config_walker_vanilla_ppo.json adv_ppo_walker/agents $semaphorename
scan_exp_folder config_halfcheetah_vanilla_ppo.json adv_ppo_halfcheetah/agents $semaphorename
scan_exp_folder config_humanoid_vanilla_ppo.json adv_ppo_humanoid/agents $semaphorename

# To stop all running process:
# killall perl; killall perl

# wait for all attacks done.
sem --wait --semaphorename $semaphorename

