from abc import abstractmethod, ABC
import loguru
import math
from typing import List
import random
from copy import deepcopy
from fedsimu.core.algorithms.base.recordable import Recordable
import time
import numpy as np


class BaseClientSampler(Recordable):
    """

    Base client sampler, to define basic interface for a client sampler. To implement a client sampler, just implement
    function _sample_client_detail.

    """

    def __init__(self, logger: loguru._Logger = loguru.logger):
        self.records = {
            'sampler': str(self.__class__),
            'sample_time_used': []
        }
        self.logger = logger

    def sample_client(self, all_client_info: List[dict]) -> List[dict]:
        for i, client_info in enumerate(all_client_info):  # drop package with nan in it.
            if 'parameter' in client_info.keys():
                client_param = client_info['parameter']
                if (np.isnan(client_param)).any():
                    del all_client_info[i]

            elif 'logit' in client_info.keys():
                client_logit = client_info['logit']
                if (np.isnan(client_logit)).any():
                    del all_client_info[i]

        for client_info in all_client_info:
            if not isinstance(client_info, dict):
                self.logger.error(f'Client info is not a dict! Check if you have any type error. Client info '
                                  f'received: {client_info}')
                raise TypeError
        start_time = time.time()
        res = self._sample_client_detail(all_client_info)
        end_time = time.time()
        self.records['sample_time_used'].append(end_time - start_time)
        return res

    @abstractmethod
    def _sample_client_detail(self, all_client_info: List[dict]) -> List[dict]:
        """


        Args:
            all_client_info: List[dict], all client info gathered from clients.

        Returns: List[dict]
            Reduced client info, the remain info are for gather in server.
        """
        pass

    def get_record_abstract(self) -> dict:
        return deepcopy(self.records)


class AllClientSampler(BaseClientSampler):
    def __init__(self, logger: loguru._Logger = None):
        super().__init__(logger=logger)

    def _sample_client_detail(self, all_client_info: List[dict]) -> List[dict]:
        return all_client_info


class RandomClientSampler(BaseClientSampler):
    def __init__(self, ratio: float = 0.5, logger: loguru._Logger = None):
        super().__init__(logger=logger)

        if not (0 <= ratio <= 1):  # ratio value check
            self.logger.error(f'RandomClientSampler accept a ratio in [0, 1], but got {ratio} instead.')
            raise TypeError
        self.ratio = ratio
        self.records.update({'random_sampler_ratio': self.ratio})

    def _sample_client_detail(self, all_client_info: List[dict]) -> List[dict]:
        total_client_num = len(all_client_info)
        sample_client_num = math.floor(total_client_num * self.ratio)

        sample_index_list = random.sample(range(total_client_num), k=sample_client_num)
        result = []
        for i in sample_index_list:
            result.append(all_client_info[i])
        return result






