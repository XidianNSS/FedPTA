import torch.utils.data as Data
import numpy as np
from typing import List
import pandas as pd


def get_x_y_from_dataloader(dataloader: Data.DataLoader) -> (np.ndarray, np.ndarray):
    """

    :param dataloader:
    :return: np.ndarray, labels
    """
    y_list = []
    x_list = []

    for x, y in dataloader:
        x_batch = x.to('cpu').detach().numpy()
        x_list.append(x_batch)

        y_batch = y.to('cpu').detach().numpy()
        y_list.append(y_batch)

    result_x = np.concatenate(x_list)
    result_y = np.concatenate(y_list)
    assert len(result_x) == len(result_y)
    return result_x, result_y


def get_y_from_dataloader(dataloader: Data.DataLoader) -> np.ndarray:
    """

    :param dataloader:
    :return: np.ndarray, labels
    """
    result = None

    for x, y in dataloader:
        y_batch = y.to('cpu').detach().numpy()
        if result is None:
            result = y_batch
        else:
            result = np.concatenate([result, y_batch])

    return result


def get_unique_labels_from_dataloader(dataloader: Data.DataLoader) -> list:
    y = get_y_from_dataloader(dataloader)
    labels = (np.unique(y)).astype('int32')
    return labels.tolist()


def get_unique_labels_from_dataloaders(dataloaders: List[Data.DataLoader]) -> list:
    """
    Got all possible labels from dataloaders of all clients
    Args:
        dataloaders:

    Returns:

    """
    y_total = []
    for dataloader in dataloaders:
        y = get_y_from_dataloader(dataloader)
        y_total.append(y)

    y_total = np.concatenate(y_total).astype('int32')
    labels = (np.unique(y_total)).astype('int32')
    return labels.tolist()


def get_y_distribution_from_dataloaders(dataloaders: List[Data.DataLoader], names: List[str], labels: list):
    """

    Args:
        dataloaders:
        names:
        labels: possible labels

    Returns:

    """
    assert len(names) == len(dataloaders)

    result_count = np.zeros([len(dataloaders), len(labels)]).astype('int32')  # gather info each client

    for idx, dataloader in enumerate(dataloaders):
        y = get_y_from_dataloader(dataloader)
        y = y.astype('int32')
        labels_y, counts = np.unique(y, return_counts=True)
        label_idx = []
        for y in labels_y:
            label_idx.append(labels.index(y))
        result_count[idx, label_idx] = counts
    result = pd.DataFrame(result_count, index=names)
    y_labels = [f'label {i}' for i in labels]
    result.columns = y_labels

    return result
