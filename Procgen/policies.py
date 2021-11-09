import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,padding=1))

    def forward(self, x):
        inputs = x
        x = self.model(x)
        return x + inputs


class ImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs):

        super(ImpalaCNN, self).__init__()

        h, w, c = obs_space.shape

        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs+1)
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

        for _ in range(3): #calculate final shape
            h = (h+1)//2
            w = (w + 1)//2

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=h*w*32,out_features=256),
            nn.ReLU(),
            self.logits_fc)
        
        # Initialize weights of logits_fc
        

    def forward(self, x):
        assert x.ndim == 4
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        output = self.model(x)
        logits, value = output[:, :-1], output[:, -1:]
        
        #dist = torch.distributions.Categorical(logits=logits)
        #print(logits - dist.logits)
        return logits, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))
