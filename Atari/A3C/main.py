from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from model import A3Cff
from train import train, train_robust
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import time

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='S',
    help='random seed (default: none)')
parser.add_argument(
    '--workers',
    type=int,
    default=16,
    metavar='W',
    help='how many training processes to use (default: 16)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--total-frames',
    type=int,
    default=20000000,
    metavar='TS',
    help='How many frames to train for before finishing (default: 20000000)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='PongNoFrameskip-v4',
    metavar='ENV',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=[0],
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: 0)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--kappa-end',
    type=float,
    default=0.9,
    metavar='SR',
    help='final value of the variable controlling importance of standard loss (default: 0.5)')
parser.add_argument('--robust',
                   dest='robust',
                   action='store_true',
                   help='train the model to be verifiably robust')
parser.add_argument(
    '--load-path',
    default=None,
    help='Path to load a model from. By default starts training a new model')
parser.add_argument(
    '--epsilon-end',
    type=float,
    default= 1/255,
    metavar='EPS',
    help='max size of perturbation trained on')
parser.add_argument('--no-smoothed',
                   dest='smoothed',
                   action='store_false',
                   help='whether to used smoothed attack epsilon schedule')
parser.add_argument('--loss-fn',
                    type=str,
                    default= "approach_2",
                    help='Which loss_fn to use, one of [approach_1, approach_2]')

parser.set_defaults(robust=False, smoothed=True)


if __name__ == '__main__':
    args = parser.parse_args()
    assert (args.loss_fn in ('approach_1', 'approach_2'))
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
        
    if args.seed:
        torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        if args.seed:
            torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Cff(env.observation_space.shape[0], env.action_space)
    if args.load_path:
        
        saved_state = torch.load(args.load_path,
                                     map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
           
    shared_model.share_memory()

    if args.optimizer == 'RMSprop':
        optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
    if args.optimizer == 'Adam':
        optimizer = SharedAdam(
            shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    optimizer.share_memory()

    processes = []

    p = mp.Process(target=test, args=(args, shared_model, optimizer, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        if args.robust:
            p = mp.Process(target=train_robust, args=(rank, args, shared_model, optimizer, env_conf))
        else:
            p = mp.Process(target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
