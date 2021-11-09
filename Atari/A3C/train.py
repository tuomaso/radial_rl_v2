from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads, EpsilonScheduler
from model import A3Cff
from player_util import Agent
from torch.autograd import Variable

def train_robust(rank, args, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    if args.seed:
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    if args.seed:
        env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Cff(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2

    if args.smoothed:
        attack_eps_scheduler = EpsilonScheduler("smoothed", 0, (2/3)*(args.total_frames/args.num_steps), 1e-9, 
                                                args.epsilon_end, args.total_frames/args.num_steps)

    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
            
        p = optimizer.param_groups[0]['params'][0]
        step = optimizer.state[p]['step']
        if step >= (args.total_frames/args.num_steps):
            return
        
        if args.smoothed:
            epsilon = attack_eps_scheduler.get_eps(0, step)
        else:
            #increase linearly until 2/3 through training
            lin_coeff = min(1, (1.5*int(step)+1)/(args.total_frames/args.num_steps))
            epsilon = lin_coeff*args.epsilon_end
        kappa = args.kappa_end
        for step in range(args.num_steps):
            player.action_train(bound_epsilon = epsilon)
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _ = player.model(Variable(player.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()
        
        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
                
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t
            
            if args.loss_fn=='approach_1':
                if gae >= 0:
                    worst_case_loss = - player.min_log_probs[i] * Variable(gae)
                else:
                    worst_case_loss = - player.max_log_probs[i] * Variable(gae)
            elif args.loss_fn=='approach_2':
                worst_case_loss = player.w_overlaps[i]
            standard_loss = -player.log_probs[i] * Variable(gae)
            
            policy_loss += kappa*standard_loss +(1-kappa)*worst_case_loss  - 0.01 * player.entropies[i]
        
        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
                
        player.clear_actions()

        
def train(rank, args, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    if args.seed:
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    if args.seed:
        env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Cff(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        
        p = optimizer.param_groups[0]['params'][0]
        step = optimizer.state[p]['step']
        if step >= (args.total_frames/args.num_steps):
            return

        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _ = player.model(Variable(player.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
