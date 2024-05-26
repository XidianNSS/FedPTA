import torchvision
from fedsimu.benchmark.partition import MyDataset
from .model import CNN, MLP
import os


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
     ]
)


def get_train_data(root_path: str):
    root_path = os.path.abspath(root_path)
    data = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
    data.data = data.data.transpose(0, 3, 1, 2)
    result = MyDataset(data.data, data.targets, name='CIFAR10')
    return result


def get_test_data(root_path: str):
    root_path = os.path.abspath(root_path)
    data = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
    data.data = data.data.transpose(0, 3, 1, 2)
    result = MyDataset(data.data, data.targets, name='CIFAR10')
    return result
