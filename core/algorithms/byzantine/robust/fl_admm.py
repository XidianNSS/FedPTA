import os
import time

os.environ["OMP_NUM_THREADS"] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from sklearn.decomposition import PCA
from fedsimu.core.algorithms.byzantine.robust.pta_utils import tensor_unfold, tensor_fold, robust_admm, \
    min_max_normalize, softmax
from fedsimu.core.algorithms.byzantine.robust.gap_statistic_utils import gap_statistic
from fedsimu.core.algorithms.byzantine.robust.base import ByzantineRobustSampler
import loguru
from typing import List
from copy import deepcopy
from sklearn.cluster import KMeans


class AdmmRobustSampler(ByzantineRobustSampler):
    def __init__(self,
                 window_size: int = 5,
                 beta: float = 100,
                 mu: float = 1,
                 tau: float = 1,
                 admm_max_iter: int = 400,
                 admm_tol: float = 1e-3,
                 use_normalize: bool = True,
                 pca_round: str = 'all',
                 pca_dimension: int = 30,
                 record_pca: bool = True,
                 record_tsne_2d: bool = False,
                 record_tsne_3d: bool = False,
                 softmax_k: float = 10,
                 logger: loguru._Logger = loguru.logger,
                 verbose: bool = False
                 ):
        ByzantineRobustSampler.__init__(self,
                                        record_pca=record_pca,
                                        record_tsne_2d=record_tsne_2d,
                                        record_tsne_3d=record_tsne_3d,
                                        logger=logger)
        self.beta = beta
        self.mu = mu
        self.tau = tau
        self.window_size = window_size
        self.admm_max_iter = admm_max_iter
        self.admm_tol = admm_tol
        self.pca_dimension = pca_dimension
        self.pca_round = pca_round
        self.use_normalize = use_normalize
        self.softmax_k = softmax_k
        self.debug_verbose = verbose
        self.records.update({
            'admm_beta': self.beta,
            'admm_tau': self.tau,
            'admm_mu': self.mu,
            'admm_window_size': self.window_size,
            'admm_tol': self.admm_tol,
            'admm_max_iter': self.admm_max_iter,
            'admm_pca_round': self.pca_round,
            'admm_pca_dimension(default)': self.pca_dimension,
            'admm_use_normalize': self.use_normalize,
            'admm_s_result': [],
            'admm_x_result': [],
            'admm_y_origin': [],
            'admm_explain_var': [],
            'admm_explain_var_softmax': [],
            'admm_softmax_k': self.softmax_k
        })
        self.window = []

    def _sample_client_detail(self, all_client_info: List[dict]) -> List[dict]:

        client_data, client_idx = self._get_param_from_client_info(all_client_info)
        self._record_plot_data(all_client_info)
        if self._skip_round_1(all_client_info):
            return all_client_info

        self.logger.info(f'Using AdmmRobustSampler, with beta={self.beta}, tau={self.tau}, mu={self.mu}, '
                         f'window_size={self.window_size}')

        # Here we assume that in one round all clients will upload their package
        sorted_client_data = [x for _, x in sorted(zip(client_idx, client_data))]
        sorted_client_data = np.array(sorted_client_data)

        self.window.append(deepcopy(sorted_client_data))
        if len(self.window) > self.window_size:  # control window size
            del self.window[0]

        hsi_y = deepcopy(np.array(self.window))  # round, client, dimension
        hsi_y = hsi_y.transpose(2, 1, 0)  # dimension, client, round
        d = hsi_y.shape[0]
        m = hsi_y.shape[1]
        T = hsi_y.shape[2]
        assert T >= 1

        if self.use_normalize:
            hsi_y = min_max_normalize(hsi_y, axis=0)

        pca_dim = self.pca_dimension
        pca_explained_var = None
        t1 = time.time()
        if self.pca_round == 'all':
            fit_data = tensor_unfold(hsi_y, axis=0)
            valid_lim = min(fit_data.shape[0], fit_data.shape[1])  # check pca n_components
            if pca_dim is None:
                pca_dim = valid_lim
            elif self.pca_dimension < 0 or self.pca_dimension > valid_lim:
                self.logger.warning(f"Using PCA but got invalid pca dimension. "
                                    f"Using {valid_lim} instead of {self.pca_dimension}.")
                pca_dim = valid_lim

            pca = PCA(n_components=pca_dim)
            res_data = pca.fit_transform(fit_data)
            hsi_y = tensor_fold(res_data, [pca_dim, m, T], axis=0)
            pca_explained_var = pca.explained_variance_

        elif self.pca_round == 'median':
            fit_round = len(self.window) // 2
            fit_data = hsi_y[:, :, fit_round]
            valid_lim = min(fit_data.shape[0], fit_data.shape[1])  # check pca n_components
            if pca_dim is None:
                pca_dim = valid_lim
            elif self.pca_dimension < 0 or self.pca_dimension > valid_lim:
                self.logger.warning(f"Using PCA but got invalid pca dimension. "
                                    f"Using {valid_lim} instead of {self.pca_dimension}.")
                pca_dim = valid_lim

            pca = PCA(n_components=pca_dim)
            pca.fit(fit_data)
            res_data = pca.transform(tensor_unfold(hsi_y, axis=0))
            hsi_y = tensor_fold(res_data, [pca_dim, m, T], axis=0)
            pca_explained_var = pca.explained_variance_

        elif self.pca_round == 'first':
            fit_round = 0
            fit_data = hsi_y[:, :, fit_round]
            valid_lim = min(fit_data.shape[0], fit_data.shape[1])  # check pca n_components
            if pca_dim is None:
                pca_dim = valid_lim
            elif self.pca_dimension < 0 or self.pca_dimension > valid_lim:
                self.logger.warning(f"Using PCA but got invalid pca dimension. "
                                    f"Using {valid_lim} instead of {self.pca_dimension}.")
                pca_dim = valid_lim

            pca = PCA(n_components=pca_dim)
            pca.fit(fit_data)
            res_data = pca.transform(tensor_unfold(hsi_y, axis=0))
            hsi_y = tensor_fold(res_data, [pca_dim, m, T], axis=0)
            pca_explained_var = pca.explained_variance_

        elif self.pca_round == 'last':
            fit_round = -1
            fit_data = hsi_y[:, :, fit_round]
            valid_lim = min(fit_data.shape[0], fit_data.shape[1])  # check pca n_components
            if pca_dim is None:
                pca_dim = valid_lim
            elif self.pca_dimension < 0 or self.pca_dimension > valid_lim:
                self.logger.warning(f"Using PCA but got invalid pca dimension. "
                                    f"Using {valid_lim} instead of {self.pca_dimension}.")
                pca_dim = valid_lim

            pca = PCA(n_components=pca_dim)
            pca.fit(fit_data)
            res_data = pca.transform(tensor_unfold(hsi_y, axis=0))
            hsi_y = tensor_fold(res_data, [pca_dim, m, T], axis=0)
            pca_explained_var = pca.explained_variance_

        else:
            self.logger.error(f"Known pca round select way: {self.pca_round}, available values are: "
                              f"'all', 'median', 'first', 'last'.")
            raise NotImplementedError
        t2 = time.time()
        self.logger.debug(f'PCA time: {t2 - t1}')
        t1 = time.time()
        x, s = robust_admm(hsi_y, beta=self.beta, mu=self.mu, tau=self.tau, tol=self.admm_tol,
                           verbose=self.debug_verbose)
        t2 = time.time()
        self.logger.debug(f'Admm time: {t2 - t1}')
        self.records['admm_x_result'].append(deepcopy(x))
        self.records['admm_s_result'].append(deepcopy(s))
        self.records['admm_y_origin'].append(deepcopy(hsi_y))
        if pca_explained_var is None:
            raise TypeError

        softmax_var = softmax(min_max_normalize(pca_explained_var) * self.softmax_k)

        self.records['admm_explain_var'].append(deepcopy(pca_explained_var))
        self.records['admm_explain_var_softmax'].append(deepcopy(softmax_var))
        k = s.shape[0]
        m = s.shape[1]
        T = s.shape[2]
        scores = np.zeros(m)
        for i in range(m):  # client
            for j in range(k):  # dim
                scores[i] += np.sum(s[j, i, :]) * softmax_var[j]
        t1 = time.time()
        cluster_k = gap_statistic(scores)
        t2 = time.time()
        self.logger.debug(f'gap statistic time: {t2 - t1}')

        labels = None
        if cluster_k >= 2:
            kmeans = KMeans(n_clusters=2)
            labels = kmeans.fit_predict(scores.reshape(-1, 1))

        decision = labels
        decision_client_idx = sorted(client_idx)
        if decision is None:
            self.logger.warning('Decision is None due to unknown reason. Check if s = 0 in admm. '
                                'Here we assert all clients are benign')
            decision = np.zeros(len(client_data))
        if sum(decision) > len(decision) / 2:  # benign 0, byzantine 1
            for i in range(len(decision)):
                decision[i] = 1 - decision[i]

        benign_idx = []
        byzantine_idx = []
        for idx, dec in enumerate(decision):
            if dec == 0:
                # for t, client_id in enumerate(client_idx):

                benign_idx.append(client_idx.index(idx))
            else:
                byzantine_idx.append(client_idx.index(idx))

        return self._get_record_after_sample(all_client_info, benign_idx=benign_idx, byzantine_idx=byzantine_idx)
