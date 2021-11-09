import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN, relatively large 4-layer
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter width: width multiplier
class cnn_4layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=2, linear_size=256):
        super(cnn_4layer, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
        self.fc2 = nn.Linear(linear_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class cnn_4layer_LeakyRelu(nn.Module):
    def __init__(self, in_ch, in_dim, width=2, linear_size=256, alpha = 0.1):
        super(cnn_4layer_LeakyRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
        self.fc2 = nn.Linear(linear_size, 10)

        self.alpha = alpha

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), self.alpha)
        x = F.leaky_relu(self.conv2(x), self.alpha)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), self.alpha)
        x = self.fc2(x)

        return x

class mlp_2layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=1):
        super(mlp_2layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 256 * width)
        self.fc2 = nn.Linear(256 * width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class mlp_3layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=1):
        super(mlp_3layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 256 * width)
        self.fc2 = nn.Linear(256 * width, 128 * width)
        self.fc3 = nn.Linear(128 * width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class mlp_5layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=1):
        super(mlp_5layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 256 * width)
        self.fc2 = nn.Linear(256 * width, 256 * width)
        self.fc3 = nn.Linear(256 * width, 256 * width)
        self.fc4 = nn.Linear(256 * width, 128 * width)
        self.fc5 = nn.Linear(128 * width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))        
        x = self.fc5(x)
        return x

# Model can be defined as a subclass of nn.Module
class cnn_7layer_alt(nn.Module):
    def __init__(self, in_ch, in_dim, width=2, linear_size=128):
        super(cnn_7layer_alt, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 4 * width, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4 * width, 8 * width, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8 * width, 8 * width, 4, stride=2, padding=1)

        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
        self.fc2 = nn.Linear(linear_size, linear_size)
        self.fc3 = nn.Linear(linear_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Model can also be defined as a nn.Sequential
def cnn_7layer(in_ch=3, in_dim=32, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def cnn_7layer_bn(in_ch=3, in_dim=32, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def cnn_7layer_bn_imagenet(in_ch=3, in_dim=32, width=64, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        Flatten(),
        nn.Linear(25088, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,200)
    )
    return model

def cnn_6layer(in_ch, in_dim, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def cnn_6layer_(in_ch, in_dim, width=32, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//4) * (in_dim//4) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model    


if __name__ == "__main__":
    from thop import profile

    # model = cnn_7layer_bn(3, 32, 64, 512) # 151M, 17M
    model = cnn_7layer_bn_imagenet(3, 32, 64, 512) # 338M, 13M

    print(model)
    dummy = torch.randn(8, 3, 56, 56)
    print(model(dummy).shape)
    macs, params = profile(model, (torch.randn(1, 3, 56, 56),))
    print(macs / 1000000, params / 1000000)
