from fedsimu.core.algorithms.base.base_sampler import BaseClientSampler
import os
os.environ["OMP_NUM_THREADS"] = '1'
import loguru
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from copy import deepcopy
import numpy as np
import math
from typing import Union, List


class ByzantineRobustSampler(BaseClientSampler):
    def __init__(self,
                 record_pca: bool = True,
                 record_tsne_2d: bool = False,
                 record_tsne_3d: bool = False,
                 logger: loguru._Logger = loguru.logger):
        BaseClientSampler.__init__(self, logger=logger)
        self.records['sampled_attackers'] = []
        self.records['sampled_attacker_idx'] = []
        self.records['pca_plot_data'] = []
        self.records['pca_plot_client_idx'] = []
        self.records['tsne_2d_plot_data'] = []
        self.records['tsne_3d_plot_data'] = []
        self.record_pca = record_pca
        self.record_tsne_2d = record_tsne_2d
        self.record_tsne_3d = record_tsne_3d

    def _get_param_from_client_info(self, all_client_info: List[dict]) -> (list, list):
        """

        Args:
            all_client_info:

        Returns:
            (client_data, client_idx), client_data is param or logit from clients, client idx show client_id
            of client_data
        """
        use_parameter = False
        use_logit = False
        client_data = []
        client_idx = []
        for client_info in all_client_info:
            client_idx.append(client_info['client_id'])
            if 'parameter' in client_info.keys():
                use_parameter = True
            elif 'logit' in client_info.keys():
                use_logit = True
            else:
                self.logger.error('Using ByzantineRobustSampler, but found no parameter or logit in package')
                raise TypeError

        if use_parameter and use_logit:
            self.logger.warning('Using ByzantineRobustSampler, '
                                'but found both parameter and logit in package. Using parameter as input')

        if use_parameter:
            for client_info in all_client_info:
                client_data.append(deepcopy(client_info['parameter']))

        elif use_logit:
            for client_info in all_client_info:
                client_data.append(deepcopy(client_info['logit']))

        else:
            self.logger.error('Using ByzantineRobustSampler, but found no parameter or logit in package')
            raise TypeError
        assert len(client_data) == len(all_client_info)
        assert len(client_data) == len(client_idx)
        return client_data, client_idx

    def _record_plot_data(self, all_client_info: List[dict]):
        client_data, client_idx = self._get_param_from_client_info(all_client_info)
        client_nums = len(client_data)
        perplexity = 30
        if client_nums <= 5:
            perplexity = client_nums - 1
        elif client_nums >= 100:
            pass
        else:
            perplexity = math.floor(client_nums * 0.3)
        if self.record_pca:
            pca_plot_data = PCA(n_components=2).fit_transform(np.array(client_data))
            self.records['pca_plot_data'].append(deepcopy(pca_plot_data))

        if self.record_tsne_2d:
            tsne_2d_plot_data = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity) \
                .fit_transform(np.array(client_data))
            self.records['tsne_2d_plot_data'].append(deepcopy(tsne_2d_plot_data))

        if self.record_tsne_3d:
            tsne_3d_plot_data = TSNE(n_components=3, init='pca', random_state=0, perplexity=perplexity) \
                .fit_transform(np.array(client_data))
            self.records['tsne_3d_plot_data'].append(deepcopy(tsne_3d_plot_data))

        self.records['pca_plot_client_idx'].append(deepcopy(client_idx))

    def _get_record_after_sample(self,
                                 all_client_info: List[dict],
                                 benign_idx: Union[List[int], np.ndarray],
                                 byzantine_idx: Union[List[int], np.ndarray]
                                 ) -> List[dict]:
        """
        This function usually is the last step in a robust sampler
        Args:
            all_client_info:
            benign_idx:
            byzantine_idx:

        Returns: Sampled client info

        """
        result_client_info = []
        for i in benign_idx:
            result_client_info.append(all_client_info[i])

        byzantine_attackers = []
        byzantine_attacker_idx = []
        for i in byzantine_idx:
            byzantine_attackers.append(f"Client {all_client_info[i]['client_id']}")
            byzantine_attacker_idx.append(all_client_info[i]['client_id'])

        byzantine_attackers = sorted(byzantine_attackers)
        byzantine_attacker_idx = sorted(byzantine_attacker_idx)
        self.records['sampled_attackers'].append(deepcopy(byzantine_attackers))
        self.records['sampled_attacker_idx'].append(deepcopy(byzantine_attacker_idx))

        self.logger.info(f"Byzantine Robust Sampler Finished, attackers = {byzantine_attackers}.")
        return result_client_info

    def _skip_round_1(self, all_client_info: List[dict]) -> bool:
        """
        Round 1 will be ignored because no attacks will happen in round 1.
        This should be after self._get_param_from_client_info and self._record_plot_data
        Examples:
            client_data, client_idx = self._get_param_from_client_info(all_client_info)  # gen data from all info
            self._record_plot_data(all_client_info)  # record pca, tsne-2d, tsne-3d
            if self._skip_round_1(all_client_info):
                return all_client_info

        Args:
            all_client_info:

        Returns: if this is round 1

        """
        assert len(all_client_info) >= 1
        round = all_client_info[0]['round']
        if round == 1:  # skip if in round 1 (no attack). In our scheme, round==0 will print [round 1] in log.
            self.records['sampled_attackers'].append([])
            self.records['sampled_attacker_idx'].append([])
            return True
        else:
            return False

class IntegratedRobustSampler(ByzantineRobustSampler):
    """
    Warning! Integrated Robust does not use super().__init__, so you have to initialize it with
    ByzantineRobustSampler.__init__ before you call IntegratedRobustSampler.__init__
    """
    def __init__(self,
                 integrated_window_size: int = 10,
                 logger: loguru._Logger = loguru.logger):
        if not hasattr(self, 'records'):
            logger.error('Integrated Robust does not use super().__init__, so you have to initialize it with '
                         'ByzantineRobustSampler.__init__ before you call IntegratedRobustSampler.__init__')
            raise NotImplementedError
        assert integrated_window_size > 0
        self.integrated_window_size = integrated_window_size
        self.records['integrated_window_size'] = self.integrated_window_size
        self.records['integrated_decision_window'] = []
        self.records['integrated_decision_window_client_idx'] = []
        self.history_decision_window = []
        self.history_client_idx_window = []



