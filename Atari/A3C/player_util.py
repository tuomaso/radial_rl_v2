from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from ibp import network_bounds

class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        
        self.min_log_probs = []
        self.max_log_probs = []
        self.w_overlaps = []
        
        self.done = True
        self.info = None
        self.reward = 0
        self.noclip_reward = 0
        self.gpu_id = -1

    def action_train(self, bound_epsilon = None):
        if self.gpu_id >= 0:
            device = torch.device('cuda:{}'.format(self.gpu_id))
        else:
            device = torch.device('cpu')
        value, logit = self.model(Variable(self.state.unsqueeze(0)))
        prob = torch.clamp(F.softmax(logit, dim=1), 1e-6, 1)
        log_prob = torch.clamp(F.log_softmax(logit, dim=1), -30, -1e-6)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        #print(prob)
        action = prob.multinomial(1).data
        #avoid issues with zero
        
        if bound_epsilon:
            upper, lower = network_bounds(self.model.model, Variable(self.state.unsqueeze(0)),
                                          epsilon=bound_epsilon)
            upper, lower = upper[:,1:], lower[:,1:]
            onehot_action = torch.zeros(upper.shape).to(device)

            onehot_action[range(upper.shape[0]), action] = 1
            min_prob = torch.clamp(F.log_softmax(onehot_action*lower+(1-onehot_action)*upper, dim=1), -30, -1e-6)
            max_prob = torch.clamp(F.log_softmax((1-onehot_action)*lower+onehot_action*upper, dim=1), -30, -1e-6)
            
            self.max_log_probs.append(max_prob.gather(1, Variable(action)))
            self.min_log_probs.append(min_prob.gather(1, Variable(action)))
            #print(logit.shape, action.shape)
            logit_diff = torch.max(torch.zeros([1]).to(device), (logit.gather(1, action).detach()-prob.detach()))
            p_diff = torch.max(torch.zeros([1]).to(device), (prob.gather(1, action).detach()-prob.detach()))
            overlap = torch.max(upper - lower.gather(1, action) + logit_diff.detach()/2, torch.zeros([1]).to(device))
            #print(overlap.shape)
            w_overlap = torch.sum(p_diff*overlap, dim=1).mean()+1e-4*torch.ones([1]).to(device)
            self.w_overlaps.append(w_overlap)

        log_prob = log_prob.gather(1, Variable(action))
        state, self.noclip_reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        self.state = self.state.to(device)
        self.reward = max(min(self.noclip_reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.eps_len += 1
        return self

    def action_test(self):
        with torch.no_grad():
            value, logit= self.model(Variable(
                self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.noclip_reward, self.done, self.info = self.env.step(action[0])
        self.reward = max(min(self.noclip_reward, 1), -1)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self
    
    def action_test_losses(self, bound_epsilon=None):
        if self.gpu_id >= 0:
            device = torch.device('cuda:{}'.format(self.gpu_id))
        else:
            device = torch.device('cpu')

        with torch.no_grad():
            value, logit= self.model(Variable(
                self.state.unsqueeze(0)))
            prob = torch.clamp(F.softmax(logit, dim=1), 1e-6, 1)
            log_prob = torch.clamp(F.log_softmax(logit, dim=1), -30, -1e-6)
            entropy = -(log_prob * prob).sum(1)
            self.entropies.append(entropy)

            action = prob.argmax(1, keepdim=True).data

            if bound_epsilon:
                upper, lower = network_bounds(self.model.model, Variable(self.state.unsqueeze(0)),
                                                  epsilon=bound_epsilon)
                upper, lower = upper[:,1:], lower[:,1:]
                onehot_action = torch.zeros(upper.shape).to(device)
                onehot_action[range(upper.shape[0]), action] = 1
                min_prob = torch.clamp(F.log_softmax(onehot_action*lower+(1-onehot_action)*upper, dim=1), -30, -1e-6)
                max_prob = torch.clamp(F.log_softmax((1-onehot_action)*lower+onehot_action*upper, dim=1), -30, -1e-6)

                self.max_log_probs.append(max_prob.gather(1, Variable(action)))
                self.min_log_probs.append(min_prob.gather(1, Variable(action)))
                logit_diff = torch.max(torch.zeros([1]).to(device), (logit.gather(1, action).detach()-prob.detach()))
                p_diff = torch.max(torch.zeros([1]).to(device), (prob.gather(1, action).detach()-prob.detach()))
                overlap = torch.max(upper - lower.gather(1, action) + logit_diff.detach()/2, torch.zeros([1]).to(device))
                #print(overlap.shape)
                w_overlap = torch.sum(p_diff*overlap, dim=1).mean()+1e-4*torch.ones([1]).to(device)
                self.w_overlaps.append(w_overlap)

            log_prob = log_prob.gather(1, Variable(action))
            state, self.noclip_reward, self.done, self.info = self.env.step(
                action.cpu().numpy())
            self.reward = max(min(self.noclip_reward, 1), -1)
            self.state = torch.from_numpy(state).float()

            self.state.to(device)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.rewards.append(self.reward)
            self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.max_log_probs = []
        self.min_log_probs = []
        self.w_overlaps = []
        return self
