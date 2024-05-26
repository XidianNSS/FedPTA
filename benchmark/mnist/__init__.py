from fedsimu.benchmark.partition import MyDataset
import torchvision
from torchvision import transforms
import torch
from .model import CNN
import os


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])


def get_train_data(root_path: str):
    root_path = os.path.abspath(root_path)
    data = torchvision.datasets.MNIST(root=root_path, train=True, transform=transform, download=True)
    data.data = torch.unsqueeze(data.data, 1)
    result = MyDataset(data.data, data.targets, name='MNIST')
    return result


def get_test_data(root_path: str):
    root_path = os.path.abspath(root_path)
    data = torchvision.datasets.MNIST(root=root_path, train=False, transform=transform, download=True)
    data.data = torch.unsqueeze(data.data, 1)
    result = MyDataset(data.data, data.targets, name='MNIST')
    return result




