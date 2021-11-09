from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class A3Cff(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Cff, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 5, stride=1, padding=1),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=1, padding=1),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, action_space.n + 1)
                                  )
        self.train()

    def forward(self, inputs):
        x = self.model(inputs)
        value = x[:, 0:1]
        actions = x[:, 1:]

        return value, actions
