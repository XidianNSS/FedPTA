import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import torch
from fedsimu.benchmark.utils import RecommendOptim
from copy import deepcopy


class CNN(nn.Module, RecommendOptim):
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def recommended_optimizer(self) -> torch.optim.Optimizer:

        adam = torch.optim.Adam(self.parameters(), lr=0.001)
        sgd = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return adam


class MLP(nn.Module, RecommendOptim):
    def __init__(self, dim_in=3*32*32, dim_out=10):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, dim_out)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def get_embedding(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x

    def recommended_optimizer(self) -> torch.optim.Optimizer:
        sgd = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return sgd


class Resnet18(nn.Module, RecommendOptim):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18()
        resnet18.fc = nn.Linear(512, 10)
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.model = resnet18

    def forward(self, x):
        return self.model(x)

    def recommended_optimizer(self) -> torch.optim.Optimizer:
        adam = torch.optim.Adam(self.model.parameters(), lr=0.01)
        return adam


