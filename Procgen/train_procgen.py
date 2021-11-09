from collections import deque
import argparse
import os
import time
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
    parser.add_argument('--exp-name', type=str, default='trial01')
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str, default=None)

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=25_000_000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--epsilon-end', type=float, default=None)

    return parser.parse_args()


def create_venv(config, is_valid=False):
    venv = ProcgenEnv(
        num_envs=config.num_envs,
        env_name=config.env_name,
        num_levels=0 if is_valid else config.num_levels,
        start_level=0 if is_valid else config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=config.num_threads,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def rollout_one_step(agent, env, obs, steps, env_max_steps=1000):

    # Step once.
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


def train(config, agent, train_env, test_env, model_dir):

    if config.model_file is not None:
        if config.gpu >= 0:
            device = torch.device("cuda:{}".format(config.gpu))
        else:
            device = torch.device("cpu")
        agent.model.load_from_file(config.model_file, device)
        logger.info('Loaded model from {}.'.format(config.model_file))
    else:
        logger.info('Train agent from scratch.')

    train_epinfo_buf = deque(maxlen=100)
    train_obs = train_env.reset()
    train_steps = np.zeros(config.num_envs, dtype=int)

    test_epinfo_buf = deque(maxlen=100)
    test_obs = test_env.reset()
    test_steps = np.zeros(config.num_envs, dtype=int)

    nbatch = config.num_envs * config.nsteps
    n_ops_per_update = nbatch * config.nepochs / (config.batch_size)
    nupdates = config.max_steps // nbatch
    max_steps = config.max_steps // config.num_envs

    logger.info('Start training for {} steps (approximately {} updates)'.format(
        config.max_steps, nupdates))

    tstart = time.perf_counter()
    for step_cnt in range(max_steps):

        # Roll-out in the training environments.
        assert agent.training
        train_obs, train_steps, train_epinfo = rollout_one_step(
            agent=agent,
            env=train_env,
            obs=train_obs,
            steps=train_steps,
        )
        train_epinfo_buf.extend(train_epinfo)

        # Roll-out in the test environments.
        with agent.eval_mode():
            assert not agent.training
            test_obs, test_steps, test_epinfo = rollout_one_step(
                agent=agent,
                env=test_env,
                obs=test_obs,
                steps=test_steps,
            )
            test_epinfo_buf.extend(test_epinfo)

        assert agent.training
        num_ppo_updates = agent.n_updates // n_ops_per_update

        if (step_cnt + 1) % config.nsteps == 0:
            tnow = time.perf_counter()
            fps = int(nbatch / (tnow - tstart))

            logger.logkv('steps', step_cnt + 1)
            logger.logkv('total_steps', (step_cnt + 1) * config.num_envs)
            logger.logkv('fps', fps)
            logger.logkv('num_ppo_update', num_ppo_updates)
            logger.logkv('eprewmean',
                         safe_mean([info['r'] for info in train_epinfo_buf]))
            logger.logkv('eplenmean',
                         safe_mean([info['l'] for info in train_epinfo_buf]))
            logger.logkv('eval_eprewmean',
                         safe_mean([info['r'] for info in test_epinfo_buf]))
            logger.logkv('eval_eplenmean',
                         safe_mean([info['l'] for info in test_epinfo_buf]))
            train_stats = agent.get_statistics()
            for stats in train_stats:
                logger.logkv(stats[0], stats[1])
            logger.dumpkvs()

            if num_ppo_updates % config.save_interval == 0:
                model_path = os.path.join(
                    model_dir, 'model_{}.pt'.format(num_ppo_updates + 1))
                agent.model.save_to_file(model_path)
                logger.info('Model save to {}'.format(model_path))

            tstart = time.perf_counter()

    # Save the final model.
    logger.info('Training done.')
    model_path = os.path.join(model_dir, 'model_final.pt')
    agent.model.save_to_file(model_path)
    logger.info('Model save to {}'.format(model_path))


def run():
    configs = parse_args()

    # Configure logger.
    log_dir = os.path.join(
        configs.log_dir,
        configs.env_name,
        'nlev_{}_{}'.format(configs.num_levels, configs.distribution_mode),
        configs.exp_name,
    )
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])

    # Create venvs.
    train_venv = create_venv(configs, is_valid=False)
    valid_venv = create_venv(configs, is_valid=True)

    # Create policy.
    policy = ImpalaCNN(
        obs_space=train_venv.observation_space,
        num_outputs=train_venv.action_space.n,
    )

    # Create agent and train.
    optimizer = torch.optim.Adam(policy.parameters(), lr=configs.lr, eps=1e-5)
    ppo_agent = PPO(
        model=policy,
        optimizer=optimizer,
        gpu=configs.gpu,
        gamma=configs.gamma,
        lambd=configs.lam,
        value_func_coef=configs.vf_coef,
        entropy_coef=configs.ent_coef,
        update_interval=configs.nsteps * configs.num_envs,
        minibatch_size=configs.batch_size,
        epochs=configs.nepochs,
        clip_eps=configs.clip_range,
        clip_eps_vf=configs.clip_range,
        max_grad_norm=configs.max_grad_norm,
        epsilon_end=configs.epsilon_end,
        max_updates = configs.max_steps * configs.nepochs / configs.batch_size
    )
    train(configs, ppo_agent, train_venv, valid_venv, log_dir)


if __name__ == '__main__':
    run()
