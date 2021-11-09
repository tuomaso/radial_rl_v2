import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import torch

from model import DuelingCnnDQN
from environment import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from utils import read_config
from matplotlib import animation

from adv_attacks.adversary import Adversary
from adv_attacks.gradient_method import FGSM
from adv_attacks.adv_model import PytorchModel
from adv_attacks.PGD import PGDAttack
from ibp import network_bounds, subsequent_bounds

parser = argparse.ArgumentParser(description='A3C')

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
    '--load-path',
    default='trained_models/Pong_radial_dqn.pt',
    metavar='LMD',
    help='path to trained model file')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='GPU to use [-1 CPU only] (default: 0)')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--pgd-video',
    type=float,
    default=None,
    metavar='PV',
    help='whether to to produce a video of the agent performing under PGD attack with given epsilon')
parser.add_argument('--video',
                    dest='video',
                    action='store_true',
                    help = 'saves a video of standard eval run of model')
parser.add_argument('--fgsm',
                    dest='fgsm',
                    action='store_true',
                    help = 'evaluate against fast gradient sign attack')
parser.add_argument('--pgd',
                   dest='pgd',
                   action='store_true',
                   help='evaluate against projected gradient descent attack')
parser.add_argument('--gwc',
                   dest='gwc',
                   action='store_true',
                   help='whether to evaluate worst possible(greedy) outcome under any epsilon bounded attack')
parser.add_argument('--acr',
                   dest='acr',
                   action='store_true',
                   help='whether to evaluate the action certification rate of an agent')
parser.add_argument('--nominal',
                   dest='nominal',
                   action='store_true',
                   help='evaluate the agents nominal performance without any adversaries')
parser.add_argument('--q-error',
                   dest='q_error',
                   action='store_true',
                   help='evaluate the error in q-functions estimate of the reward')
parser.add_argument('--num-episodes',
                    type = int,
                    default = 20,
                    help='how many episodes to run each evaluation for')

parser.set_defaults(video=False, fgsm=False, pgd=False, gwc=False, action_pert=False, acr=False)


def record_game(curr_model, env, args):

    #env = Monitor(env, './videos/{}'.format(save_name), force=True)
    frames = []
    state = env.reset()
    if args.gpu_id >= 0:
        with torch.cuda.device(args.gpu_id):
            curr_model = curr_model.cuda()
    frames.append(env.render(mode="rgb_array"))
    episode_reward = 0
    
    with torch.no_grad():
        for _ in range(args.max_episode_length):
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            value, advs = curr_model.forward(input_x)
            output = value + advs
            action = torch.argmax(output, dim=1)
            next_state, reward, done, info = env.step(action[0])    
            frames.append(env.render(mode="rgb_array"))
            episode_reward += reward
            state = next_state
            
            if done:
                return episode_reward, np.array(frames, dtype=np.uint8)
        
        return episode_reward, np.array(frames, dtype=np.uint8)

            
def attack_eval(curr_model, env, args, epsilon=1e-4, attack_type='FGSM', record=False):
    assert attack_type in ('FGSM', 'PGD'), 'Invalid attack type'
    loss_func = torch.nn.CrossEntropyLoss()
    m = PytorchModel(curr_model, loss_func,(0, 1), channel_axis=1, nb_classes=env.action_space, device=args.gpu_id)
    
    if attack_type=='FGSM':
        attack = FGSM(m)
        attack_config = {"epsilons": [epsilon], 'steps': 1}
    elif attack_type == 'PGD':
        attack = PGDAttack(m)
        attack_config = {"epsilon": epsilon, "steps": 10, "relative_step_size":0.1}
        
    total_count = 0
    fooling_count = 0
    
    episode_reward = 0
    state = env.reset()

    if record:
        frames = []
        frames.append(env.render(mode="rgb_array"))
    
    for _ in range(args.max_episode_length):
        total_count += 1
        input_x = torch.FloatTensor(state).unsqueeze(0)
        if args.gpu_id >= 0:
            with torch.cuda.device(args.gpu_id):
                input_x = input_x.cuda()
        value, advs = curr_model.forward(input_x)
        output = value + advs

        action = torch.argmax(output, dim=1)
        inputs, labels= input_x.detach().cpu().numpy(), action.detach().cpu().numpy()
        adversary = Adversary(inputs, labels[0])
        adversary = attack(adversary, **attack_config)
        
        if adversary.is_successful():
            fooling_count += 1
            next_state, reward, done, info = env.step(adversary.adversarial_label)
        else:
            next_state, reward, done, info = env.step(action[0])
        
        episode_reward += reward
        state = next_state

        if record:
            frames.append(env.render(mode="rgb_array"))

        if done:
            break
    
    print("{}: fooling_count={}, total_count={}, fooling_rate={:.3f}".format(
        attack_type, fooling_count, total_count, float(fooling_count) / total_count))
    print('Reward under {} attack {}'.format(attack_type, episode_reward))
    
    if record:
        return episode_reward, frames
    else:
        return episode_reward


def eval_greedy_wc(curr_model, env, args, epsilon=1e-4):
    episode_reward = 0
    state = env.reset()
    
    #make sure mean of advantage is zero
    with torch.no_grad():
        curr_model.advantage[-1].weight[:] -= torch.mean(curr_model.advantage[-1].weight, dim=0, keepdim=True)
        curr_model.advantage[-1].bias[:] -= torch.mean(curr_model.advantage[-1].bias, dim=0, keepdim=True)

    with torch.no_grad():
        for _ in range(args.max_episode_length):
            state = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    state = state.cuda()
            value, advs = curr_model.forward(state)
            output = value + advs

            upper, lower = network_bounds(curr_model.cnn, state, epsilon)
            upper, lower = subsequent_bounds(curr_model.advantage, upper, lower)
            upper += value
            lower += value

            impossible = upper < torch.max(lower, dim=1)[0]
            #add a large number to ignore impossible ones, choose possible action with smallest q-value
            worst_case_action = torch.argmin(output+1e6*impossible, dim=1)
            next_state, reward, done, info = env.step(worst_case_action[0])
            episode_reward += reward
            state = next_state
            if done:
                break
        state = env.reset()
        print('Worst case reward {}'.format(episode_reward))
        return episode_reward

def eval_action_cert_rate(curr_model, env, args, epsilon=1e-4):
    episode_reward = 0
    state = env.reset()
    total = 0
    certified = 0
    #make sure mean of advantage is zero
    with torch.no_grad():
        curr_model.advantage[-1].weight[:] -= torch.mean(curr_model.advantage[-1].weight, dim=0, keepdim=True)
        curr_model.advantage[-1].bias[:] -= torch.mean(curr_model.advantage[-1].bias, dim=0, keepdim=True)

    with torch.no_grad():
        for _ in range(args.max_episode_length):
            state = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    state = state.cuda()
            value, advs = curr_model.forward(state)
            output = value + advs
            action = torch.argmax(output, dim=1)
            

            upper, lower = network_bounds(curr_model.cnn, state, epsilon)
            upper, lower = subsequent_bounds(curr_model.advantage, upper, lower)
            upper += value
            lower += value

            #remove the action selected from calculations
            upper[:, action] = -1e10
            
            max_other = torch.max(upper, dim=1)[0]
            if lower[:, action] > max_other:
                certified += 1
            total += 1

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        state = env.reset()
        print('Reward:{}, action certification rate {:.4f}'.format(episode_reward, certified/total))
        return certified/total       

def eval_q_error(curr_model, env, args, save_name, seed, epsilon=0.0):
    episode_reward = 0
    state = env.reset()
    q_value_seq = []
    reward_seq = []
    with torch.no_grad():
        for _ in range(args.max_episode_length):
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            value, advs = curr_model.forward(input_x)
            output = value + advs
            if random.random() < epsilon:
                action = random.randint(0, output.shape[1]-1)
            else:
                action = torch.argmax(output[0])
            q_value_seq.append(float(output[0,action].cpu()))

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            reward_seq.append(max(-1, min(1,reward)))
            state = next_state
            if done:
                break
    state = env.reset()

    reward_seq.reverse()
    cumulative_rewards = []
    #calculate rewards in inverse order for ease, flip back later
    for i, reward in enumerate(reward_seq):
        if i ==0:
            cumulative_rewards.append(reward)
        else:
            cumulative_rewards.append(reward + args.gamma*cumulative_rewards[-1])
    cumulative_rewards.reverse()
    print(episode_reward)
    errors = np.array(q_value_seq)-np.array(cumulative_rewards)
    plt.plot(errors[:250], label='q-value-error')
    np.save('figures/{}/q_error_{}.npy'.format(save_name, seed), errors)
    
    plt.legend()
    plt.xlabel('Step')
    plt.savefig('figures/{}/q_error_{}.png'.format(save_name, seed))
    plt.close()
    
    print('Avg Q error:{}, action perturbation:{}'.format(np.mean(errors), epsilon))
    print('Avg abs(Q error):{}, action perturbation:{}'.format(np.mean(np.abs(errors)), epsilon))
    return np.mean(errors), np.mean(np.abs(errors))

def eval_action_pert(curr_model, env, args, epsilon=0.01):
    episode_reward = 0
    state = env.reset()

    with torch.no_grad():
        for _ in range(args.max_episode_length):
            input_x = torch.FloatTensor(state).unsqueeze(0)
            if args.gpu_id >= 0:
                with torch.cuda.device(args.gpu_id):
                    input_x = input_x.cuda()
            value, advs = curr_model.forward(input_x)
            output = value + advs
            
            if random.random() < epsilon:
                action = random.randint(0, output.shape[1]-1)
            else:
                action = torch.argmax(output[0])
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        state = env.reset()
        print('Reward under {} action perturbation:{}'.format(epsilon, episode_reward))
        return episode_reward

def plot_results(epsilons, rewards, save_name, eval_type):
    rewards = np.sort(rewards, axis=1)
    plt.plot(epsilons, np.mean(rewards, axis=1), label='mean')
    plt.fill_between(epsilons, rewards[:, -1], rewards[:, 0], alpha=0.2, label='interval')
    plt.legend()
    plt.xlabel('l-inf perturbation')
    plt.ylabel('reward')
    #plt.xscale('log')
    np.save('figures/{}/{}.npy'.format(save_name, eval_type), rewards)
    plt.savefig('figures/{}/{}.png'.format(save_name, eval_type))
    plt.close()

def set_seed(random_seed):
    #set seeds for reproducible results
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    env.action_space.seed(random_seed)

if __name__ == '__main__':
    args = parser.parse_args()
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

    model = DuelingCnnDQN(env.observation_space.shape[0], env.action_space)
    
    if args.gpu_id >= 0:
        weights = torch.load(args.load_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
        if "model_state_dict" in weights.keys():
            weights = weights['model_state_dict']
        model.load_state_dict(weights)
        
        with torch.cuda.device(args.gpu_id):
            model.cuda()
    else:
        weights = torch.load(args.load_path, map_location=torch.device('cpu'))
        if "model_state_dict" in weights.keys():
            weights = weights['model_state_dict']
        model.load_state_dict(weights)
    model.eval()
    
    
    save_name = (args.load_path.split('/')[-1]).split('.')[0]
    if not os.path.exists('videos'):
        os.mkdir('videos')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    if not os.path.exists('figures/'+save_name):
        os.mkdir('figures/'+save_name)
            
    if args.video:
        reward, frames = record_game(model, env, args)
        print(reward)
        frames = [frame for i,frame in enumerate(frames) if i%3==0]
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save('videos/'+save_name+'_nominal.gif', writer='imagemagick', fps=20)
    
    if args.pgd_video:
        reward, frames = attack_eval(model, env, args, args.pgd_video, 'PGD', record=True)
        print(reward)
        frames = [frame for i,frame in enumerate(frames) if i%3==0]
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save('videos/'+save_name+'_pgd_{}.gif'.format(args.pgd_video), writer='imagemagick', fps=20)
    
    epsilons = [1/255, 3/255, 5/255]
    if args.fgsm:
        np.save('figures/{}/fgsm_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(args.num_episodes):
                set_seed(i)
                reward = attack_eval(model, env, args, epsilon, 'FGSM')
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        plot_results(epsilons, rewards, save_name, 'fgsm')
        
        
    if args.pgd:
        np.save('figures/{}/pgd_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(args.num_episodes):
                set_seed(i)
                reward = attack_eval(model, env, args, epsilon, 'PGD')
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        
        plot_results(epsilons, rewards, save_name, 'pgd')
        
    if args.gwc:
        np.save('figures/{}/greedy_wc_epsilons.npy'.format(save_name), epsilons)
        rewards = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rewards = []
            for i in range(args.num_episodes):
                set_seed(i)
                reward = eval_greedy_wc(model, env, args, epsilon)
                curr_rewards.append(reward)
            rewards.append(curr_rewards)
        
        plot_results(epsilons, rewards, save_name, 'greedy_wc')
        
    if args.acr:
        np.save('figures/{}/acr_epsilons.npy'.format(save_name), epsilons)
        rates = []
        for epsilon in epsilons:
            print(epsilon)
            curr_rates = []
            for i in range(args.num_episodes):
                set_seed(i)
                rate = eval_action_cert_rate(model, env, args, epsilon)
                curr_rates.append(rate)
            rates.append(curr_rates)
        
        plot_results(epsilons, rates, save_name, 'acr')
        
    if args.nominal:
        curr_rewards = []
        for i in range(args.num_episodes):
            set_seed(i)
            reward = eval_action_pert(model, env, args, epsilon=0)
            curr_rewards.append(reward)
        rewards = np.sort(curr_rewards)
        plt.hist(rewards, bins=10)
        plt.title('Nominal mean reward:{:.1f}'.format(np.mean(rewards)))
        np.save('figures/{}/nominal.npy'.format(save_name), rewards)
        plt.savefig('figures/{}/nominal.png'.format(save_name))
        plt.close()
    
    if args.q_error:
        mean_errors = []
        abs_errors = []
        for i in range(args.num_episodes):
            set_seed(i)
            mean_error, abs_error = eval_q_error(model, env, args, save_name, i, epsilon=0)
            mean_errors.append(mean_error)
            abs_errors.append(abs_error)
        print("Average error:{:.4f}, Abs(error):{:.4f}".format(np.mean(mean_errors), np.mean(abs_errors)))
    
