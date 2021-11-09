import torch
import torch.nn as nn
import torch.nn.functional as F
from policies import ResidualBlock

def initial_bounds(x0, epsilon):
    '''
    x0 = input, b x c x h x w
    '''
    upper = x0+epsilon
    lower = x0-epsilon
    return upper, lower

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, x0, epsilon):
    '''
    get interval bound progation upper and lower bounds for the activation of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0, epsilon)
    return subsequent_bounds(model, upper, lower)

def subsequent_bounds(model, upper, lower):
    '''
    get interval bound progation upper and lower bounds for the activation of a model,
    given bounds of the input
    
    model: a nn.Sequential module
    upper: upper bound on input layer, b x input_shape
    lower: lower bound on input layer, b x input_shape
    '''
    #print(model._modules)
    for layer in model:
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Linear, nn.Conv2d):
            upper, lower = weighted_bound(layer, upper, lower)
        elif type(layer)==ResidualBlock:
            new_upper, new_lower = subsequent_bounds(layer.model, upper, lower)
            upper, lower = new_upper+upper, new_lower+lower
        else:
            print('Unsupported layer:', type(layer))
    return upper, lower
