import torch.nn.functional as F
import torch
from fedsimu.benchmark.utils import RecommendOptim


class CNN(torch.nn.Module, RecommendOptim):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(7 * 7 * 5, 100)
        self.fc2 = torch.nn.Linear(100, 62)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 5)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

    def recommended_optimizer(self) -> torch.optim.Optimizer:
        adam = torch.optim.Adam(self.parameters(), lr=0.001)
        sgd = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return adam
