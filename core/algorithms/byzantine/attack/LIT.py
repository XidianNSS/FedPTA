from .attack_container import BaseAttackerContainer
from .data_poisoning import BackdoorAttacker
from fedsimu.core.algorithms.base.base_client import BaseClient
from fedsimu.core.algorithms.base.base_attacker import BaseAttacker
from typing import List
import loguru
from copy import deepcopy
import numpy as np
from scipy.stats import norm
import math


class LITAttacker(BackdoorAttacker):
    def local_process_benign(self, global_info: dict):
        self.logger.info("[Round {}]LIT Benign simu process. Start training on client {}."
                         .format(global_info['round'] + 1,
                                 self.client_id))
        self.log_client_abstract()

        self.uplink_package = {'client_id': self.client_id, 'round': global_info['round'] + 1}
        self.before_train(global_info, record_info=False)  # Usually load global info from server.
        self.train(global_info, record_info=False)  # Usually train process
        self.after_train(global_info, record_info=False)  # Usually upload local info to server.
        self.record_model(global_info)
        return self.uplink_package


class LITAttackerContainer(BaseAttackerContainer):
    def __init__(self,
                 clients: List[BaseClient] = None,
                 n: int = None,
                 m: int = None,
                 logger: loguru._Logger = loguru.logger
                 ):
        """

        Args:
            clients:
            n: int, client num of whole experiment. if None, n=len(self.clients)
            m: int, byzantine client num of whole experiment. if None, m=num of by clients in self.clients
            logger:
        """
        BaseAttackerContainer.__init__(self,
                                       clients=clients,
                                       logger=logger)

        self.records['attacker_container'] = True
        if n is not None:
            assert n > 0
        if m is not None:
            assert m > 0
        if n is not None and m is not None:
            assert n >= m

        self.n = n
        self.m = m

    def download(self, package: dict):

        self.uplink_package = {'packages': []}  # refresh cache
        server_package = package['packages']
        byzantine_benign_simu_packages = []

        for client in self.clients:
            if not isinstance(client, LITAttacker):  # gather real benign info
                self.logger.info(f'Running benign client {client.client_id} in LIT client container.')
                client_package = client.local_process(server_package)
                self.uplink_package['packages'].append(client_package)
            else:  # gather fake benign info
                self.logger.info(f'Running LIT client {client.client_id}(benign simu) in LIT client container.')
                byzantine_benign_simu_packages.append(client.local_process_benign(server_package))

        if len(byzantine_benign_simu_packages) == 0:
            self.logger.warning('LIT Container should have at least one LITAttacker. Got 0.')

        if self.n is None:
            self.n = len(self.clients)
        if self.m is None:
            self.m = len(byzantine_benign_simu_packages)

        if 'parameter' not in byzantine_benign_simu_packages[0].keys():
            self.logger.error("LIT Only support FedAvg like FL setting, which require 'parameter' in uplink package.")
            raise TypeError

        simu_parameters = []
        for simu_package in byzantine_benign_simu_packages:
            simu_parameters.append(deepcopy(simu_package['parameter']))

        param_mean = np.mean(simu_parameters, axis=0)
        param_std_dev = np.std(simu_parameters, axis=0, ddof=1)

        server_package['avg_parameter'] = deepcopy(param_mean)

        s = math.floor(self.n / 2 + 1) - self.m
        z = norm.ppf((self.n - self.m - s) / (self.n - self.m))

        for client in self.clients:
            if isinstance(client, LITAttacker):
                self.logger.info(f'Running LIT client {client.client_id}(attack) in LIT client container.')
                attacked_package = client.local_process(server_package)

                attacked_package['parameter'] = \
                    np.clip(attacked_package['parameter'],
                            param_mean - z * param_std_dev,
                            param_mean + z * param_std_dev)

                self.uplink_package['packages'].append(attacked_package)

        self.logger.info('LIT Client container process success. Package ready for uploading.')




