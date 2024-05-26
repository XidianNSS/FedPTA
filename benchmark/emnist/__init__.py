import torchvision.transforms as transforms
import torchvision
from fedsimu.benchmark.partition import MyDataset
import torch
from .model import CNN


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_train_data(root_path: str, spilt: str = 'byclass'):
    data = torchvision.datasets.EMNIST(root=root_path, split=spilt, train=True, download=True, transform=transform)
    data.data = torch.unsqueeze(data.data, 1)
    result = MyDataset(data.data, data.targets, name='EMNIST')
    return result


def get_test_data(root_path: str, spilt: str = 'byclass'):
    data = torchvision.datasets.EMNIST(root=root_path, split=spilt, train=False, download=True, transform=transform)
    data.data = torch.unsqueeze(data.data, 1)
    result = MyDataset(data.data, data.targets, name='EMNIST')
    return result


if __name__ == '__main__':
    data = get_train_data('E:/workspace/datasets')
    temp = str(data)
    print()


