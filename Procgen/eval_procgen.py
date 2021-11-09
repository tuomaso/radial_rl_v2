from collections import deque
import argparse
import os
import time, datetime
import torch
import numpy as np

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from policies import ImpalaCNN
from ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env-name', type=str, default='fruitbot')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=200)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str, default=None)

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--nsteps', type=int, default=512)

    parser.add_argument('--standard', dest='standard', action='store_true', help='evaluate standard performance')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='makes agent always big most likely action')
    parser.add_argument('--pgd', dest='pgd', action='store_true', help='evaluate under PGD attack')
    parser.add_argument('--gwc', dest='gwc', action='store_true', help='evaluate Greedy Worst-Case Reward')
    parser.set_defaults(deterministic=False, pgd=False, standard=False, gwc=False)
    return parser.parse_args()


def create_venv(config, is_valid=False, seed=None):
    venv = ProcgenEnv(
        num_envs=config.num_envs,
        env_name=config.env_name,
        num_levels=0 if is_valid else config.num_levels,
        start_level=0 if is_valid else config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=config.num_threads,
        rand_seed = seed
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def rollout_one_step(agent, env, obs, steps, env_max_steps=1000, pgd_epsilon=None, gwc_epsilon=None):
    assert not (pgd_epsilon and gwc_epsilon)
    # Step once.
    if pgd_epsilon:
        action = agent.batch_act_pgd(obs, pgd_epsilon)
    elif gwc_epsilon:
        action = agent.batch_act_gwc(obs, gwc_epsilon)
    else:
        action = agent.batch_act(obs)
    new_obs, reward, done, infos = env.step(action)
    steps += 1
    reset = steps == env_max_steps
    steps[done] = 0

    # Save experience.
    agent.batch_observe(
        batch_obs=new_obs,
        batch_reward=reward,
        batch_done=done,
        batch_reset=reset,
    )

    # Get rollout statistics.
    epinfo = []
    for info in infos:
        maybe_epinfo = info.get('episode')
        if maybe_epinfo:
            epinfo.append(maybe_epinfo)

    return new_obs, steps, epinfo


def evaluate(config, agent, train_env, test_env, pgd_epsilon=None, gwc_epsilon=None):

    assert not (pgd_epsilon and gwc_epsilon)
    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
    else:
        device = torch.device("cpu")
    agent.model.load_from_file(config.model_file, device)
    logger.info('Loaded model from {}.'.format(config.model_file))

    train_epinfo_buf = []#deque(maxlen=1000)
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    test_epinfo_buf = []#deque(maxlen=1000)
    test_obs = test_env.reset()
    test_steps = np.zeros(config.num_envs, dtype=int)

    tstart = time.perf_counter()

    step_cnt = 0
    while len(train_epinfo_buf)<config.num_episodes or len(test_epinfo_buf)<config.num_episodes:
        step_cnt += 1
        # Roll-out in the test environments.
        with agent.eval_mode():
            assert not agent.training
            if len(train_epinfo_buf)<config.num_episodes:
                train_obs, train_steps, train_epinfo = rollout_one_step(
                    agent=agent,
                    env=train_env,
                    obs=train_obs,
                    steps=train_steps,
                    pgd_epsilon=pgd_epsilon,
                    gwc_epsilon=gwc_epsilon
                )
                train_epinfo_buf.extend(train_epinfo)

            if len(test_epinfo_buf)<config.num_episodes:
                test_obs, test_steps, test_epinfo = rollout_one_step(
                    agent=agent,
                    env=test_env,
                    obs=test_obs,
                    steps=test_steps,
                    pgd_epsilon=pgd_epsilon,
                    gwc_epsilon=gwc_epsilon
                )
                test_epinfo_buf.extend(test_epinfo)

        if (step_cnt + 1) % config.nsteps == 0:
            
            tnow = time.perf_counter()
            time_taken = tnow - tstart
            logger.logkv('time', time_taken)
            logger.logkv('num_episodes',
                            len(train_epinfo_buf[:config.num_episodes]))
            logger.logkv('eprewmean',
                            safe_mean([info['r'] for info in train_epinfo_buf[:config.num_episodes]]))
            logger.logkv('eplenmean',
                            safe_mean([info['l'] for info in train_epinfo_buf[:config.num_episodes]]))
            logger.logkv('num_eval_episodes',
                            len(test_epinfo_buf[:config.num_episodes]))
            logger.logkv('eval_eprewmean',
                            safe_mean([info['r'] for info in test_epinfo_buf[:config.num_episodes]]))
            logger.logkv('eval_eplenmean',
                            safe_mean([info['l'] for info in test_epinfo_buf[:config.num_episodes]]))
            logger.dumpkvs()

    tnow = time.perf_counter()
    time_taken = tnow - tstart
    logger.logkv('time', time_taken)
    logger.logkv('num_episodes',
                            len(train_epinfo_buf[:config.num_episodes]))
    logger.logkv('eprewmean',
                    safe_mean([info['r'] for info in train_epinfo_buf[:config.num_episodes]]))
    logger.logkv('eplenmean',
                    safe_mean([info['l'] for info in train_epinfo_buf[:config.num_episodes]]))
    logger.logkv('num_eval_episodes',
                            len(test_epinfo_buf[:config.num_episodes]))
    logger.logkv('eval_eprewmean',
                    safe_mean([info['r'] for info in test_epinfo_buf[:config.num_episodes]]))
    logger.logkv('eval_eplenmean',
                    safe_mean([info['l'] for info in test_epinfo_buf[:config.num_episodes]]))
    logger.dumpkvs()
    train_rewards = [info['r'] for info in train_epinfo_buf[:config.num_episodes]]
    eval_rewards = [info['r'] for info in test_epinfo_buf[:config.num_episodes]]
    return safe_mean(train_rewards), np.std(train_rewards)/len(train_rewards)**0.5, safe_mean(eval_rewards), np.std(eval_rewards)/len(eval_rewards)**0.5

def run():
    configs = parse_args()

    # Configure logger.
    log_dir = os.path.join(
        *(configs.model_file.split('/')[:-1]), 'eval'
    )
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])

    result_filename = os.path.join(log_dir, 'results_{}.txt'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    results = open(result_filename, 'w')

    random_seed = 0
    # Create venvs.
    train_venv = create_venv(configs, is_valid=False, seed=random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create policy.
    policy = ImpalaCNN(
        obs_space=train_venv.observation_space,
        num_outputs=train_venv.action_space.n,
    )

    # Create agent and train.
    optimizer = torch.optim.Adam(policy.parameters(), lr=0, eps=1e-5)
    ppo_agent = PPO(
        model=policy,
        optimizer=optimizer,
        gpu=configs.gpu,
        minibatch_size=configs.batch_size,
        act_deterministically = configs.deterministic
    )
    
    if configs.standard:
        train_venv = create_venv(configs, is_valid=False, seed=random_seed)
        valid_venv = create_venv(configs, is_valid=True, seed=random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        train_mean, train_std, eval_mean, eval_std = evaluate(configs, ppo_agent, train_venv, valid_venv)
        out = "Standard, Deterministic: {}, Train: {:.4f} +- {:.4f}, Eval: {:.4f} +- {:.4f} \n".format(configs.deterministic, train_mean, train_std, eval_mean, eval_std)
        print(out)
        results.write(out)
    
    if configs.gwc:
        train_venv = create_venv(configs, is_valid=False, seed=random_seed)
        valid_venv = create_venv(configs, is_valid=True, seed=random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        epsilon = 1
        train_mean, train_std, eval_mean, eval_std = evaluate(configs, ppo_agent, train_venv, valid_venv, gwc_epsilon=epsilon)
        out = "GWC {}/255, Train: {:.4f} +- {:.4f}, Eval: {:.4f} +- {:.4f} \n".format(epsilon, train_mean, train_std, eval_mean, eval_std)
        print(out)
        results.write(out)

    if configs.pgd:
        epsilons = [1, 3, 5]
        for pgd_epsilon in epsilons:
            train_venv = create_venv(configs, is_valid=False, seed=random_seed)
            valid_venv = create_venv(configs, is_valid=True, seed=random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            train_mean, train_std, eval_mean, eval_std = evaluate(configs, ppo_agent, train_venv, valid_venv, pgd_epsilon=pgd_epsilon)
            out = "PGD {}/255, Deterministic: {}, Train: {:.4f} +- {:.4f}, Eval: {:.4f} +- {:.4f} \n".format(pgd_epsilon, configs.deterministic, train_mean, train_std, eval_mean, eval_std)
            print(out)
            results.write(out)

    results.close()

if __name__ == '__main__':
    run()
