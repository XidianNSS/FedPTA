import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
from typing import List
from copy import deepcopy
warnings.filterwarnings('ignore', category=FutureWarning)


def gap_statistic(data: np.ndarray, k_values: List[int] = range(1, 11)):
    """
    Compute Gap Statistics values

    Parameters:
        data: numpy array, input data
        k_values: list, a list containing the number of clusters to try
        n_init: int, optional, number of times the k-means algorithm will be run with different centroid seeds.

    Returns:
        gap_values: list, containing the Gap Statistics values for each k value
    """
    data = deepcopy(data)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    gap_values = []
    for k in k_values:

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(scaled_data)

        wk = kmeans.inertia_
        random_data = np.random.rand(*data.shape)
        random_data = scaler.transform(random_data)

        random_kmeans = KMeans(n_clusters=k)
        random_kmeans.fit(random_data)

        wk_random = random_kmeans.inertia_
        gap = np.log(wk_random) - np.log(wk)
        gap_values.append(gap)

    temp = sorted(zip(gap_values, k_values))[::-1]
    gap_v = temp[0][0]
    max_gap_k = temp[0][1]
    return max_gap_k





