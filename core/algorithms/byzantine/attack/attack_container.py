from fedsimu.core.container.base_container import BasicClientContainer
from fedsimu.core.algorithms.base.base_client import BaseClient
from typing import List
import loguru


class BaseAttackerContainer(BasicClientContainer):
    """
    Attacker Containers are for special attacks like LIT attack, which will use all client param to do
    """
    def __init__(self,
                 clients: List[BaseClient] = None,
                 logger: loguru._Logger = loguru.logger
                 ):
        super().__init__(clients=clients, logger=logger)




