import loguru
from fedsimu.core.algorithms.base.recordable import Recordable
from fedsimu.core.container.base_container import BaseContainer
from abc import abstractmethod, ABC
from typing import List


class BaseRecorder(object):
    def __init__(self,
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger

    @abstractmethod
    def save_experiment(self,
                        container_records: List[dict],
                        name: str = None,
                        additional_info: dict = {}):
        """

        A saved object(dict) should at least have:
            'server_info': server_record,
            'client_info': [client_record_1, client_record_2, ...],
            'name': experiment_name

        """
        raise NotImplementedError

    @abstractmethod
    def load_experiment(self) -> dict:
        raise NotImplementedError



