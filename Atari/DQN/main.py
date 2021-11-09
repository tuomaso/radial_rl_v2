from __future__ import print_function, division
import os
import argparse
import torch
from environment import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from utils import read_config
from model import DuelingCnnDQN
from train import train

#undo_logger_setup()
parser = argparse.ArgumentParser(description='DQN')
parser.add_argument(
    '--lr',
    type=float,
    default=0.000125,
    metavar='LR',
    help='learning rate (default: 0.000125)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='S',
    help='random seed (default: None)')
parser.add_argument(
    '--total-frames',
    type=int,
    default=6000000,
    metavar='TS',
    help='How many frames to train with')
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
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='optimizer to use, one of {Adam, RMSprop}')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=False,
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
    default=0.8,
    metavar='SR',
    help='final value of the variable controlling importance of standard loss (default: 0.5)')
parser.add_argument('--robust',
                   dest='robust',
                   action='store_true',
                   help='train the model to be verifiably robust')
parser.add_argument(
    '--load-path',
    type=str,
    default=None,
    help='Path to load a model from. By default starts training a new model, use this for loading only model params')
parser.add_argument(
    '--resume-training',
    type=str,
    default=None,
    help='''Path to resume training from. This argument also loads optimizer params and training step in addition to model.
         Dont use together with load-path''')

parser.add_argument(
    '--attack-epsilon-end',
    type=float,
    default=1/255,
    metavar='EPS',
    help='max size of perturbation trained on')
parser.add_argument(
    '--attack-epsilon-schedule',
    type=int,
    default=4000000,
    help='The frame by which to reach final perturbation')
parser.add_argument(
    '--exp-epsilon-end',
    type=float,
    default=0.01,
    help='for epsilon-greedy exploration')
parser.add_argument(
    '--exp-epsilon-decay',
    type=int,
    default=500000,
    help='controls linear decay of exploration epsilon')
parser.add_argument(
    '--replay-initial',
    type=int,
    default=50000,
    help='How many frames of experience to collect before starting to learn')
parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    help='Batch size for updating agent')
parser.add_argument(
    '--updates-per-frame',
    type=int,
    default=32,
    help='How many gradient updates per new frame')
parser.add_argument(
    '--buffer-size',
    type=int,
    default=200000,
    help='How frames to store in replay buffer')

parser.add_argument(
    '--loss-fn',
    type=str,
    default = 'approach_2',
    help='Which loss function to use for robust training, one of [approach_1, approach_2]'
)
parser.add_argument('--decay-zero',
                   dest='decay_zero',
                   action='store_true',
                   help='whether to decay exploration epsilon down to zero')
parser.add_argument('--no-smoothed',
                   dest='smoothed',
                   action='store_false',
                   help='whether to use linear attack epsilon schedule instead of default smoothed one')
parser.add_argument("--adam-eps",
                    type = float,
                    default = 1e-8,
                    help = 'the epsilon parameter for adam optimizer')
parser.add_argument('--linear-kappa',
                   dest='constant_kappa',
                   action='store_false',
                   help='whether to use a linear kappa schedule instead of default constant kappa')

parser.set_defaults(robust=False, decay_zero=False, smoothed=True, constant_kappa = True)

if __name__ == '__main__':
    args = parser.parse_args()
    assert (args.loss_fn in ('approach_1', 'approach_2'))
    if args.seed:
        torch.manual_seed(args.seed)
        if args.gpu_id>=0:
            torch.cuda.manual_seed(args.seed)

    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    if "NoFrameskip" not in args.env:
        env = make_atari_cart(args.env)
    else:
        env = make_atari(args.env)
        env = wrap_deepmind(env, central_crop=True, clip_rewards=False, episode_life=False, **env_conf)
        env = wrap_pytorch(env)
    
    curr_model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
    target_model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
        
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    
    if args.load_path:
        saved_state = torch.load(
            args.load_path,
            map_location=lambda storage, loc: storage)
        if "model_state_dict" in saved_state.keys():
            saved_state = saved_state['model_state_dict']
        curr_model.load_state_dict(saved_state)
        
    target_model.load_state_dict(curr_model.state_dict())
    if args.gpu_id >= 0:
        with torch.cuda.device(args.gpu_id):
            curr_model.cuda()
            target_model.cuda()
    
    train(curr_model, target_model, env, args)
        
