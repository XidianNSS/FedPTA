from fedsimu.core.algorithms.fedavg import FedAvgClient
from fedsimu.core.algorithms.byzantine.attack.data_poisoning import RandomLabelFlippingAttacker, \
    BackdoorAttacker, ScalingBackdoorAttacker
from fedsimu.core.algorithms.byzantine.attack.model_attack import GaussianNoiseAttacker
from fedsimu.core.algorithms.byzantine.attack.LIT import LITAttacker
import torch
import numpy as np
import loguru
from typing import List
import torch.utils.data as Data


class FedAvgLFClient(RandomLabelFlippingAttacker, FedAvgClient):
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 dataloader: Data.DataLoader,
                 device: str = 'cpu',
                 logger: loguru._Logger = loguru.logger,
                 local_epoch: int = 10,
                 local_lr: float = 0.001,
                 optimizer: str = 'default',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 sgd_momentum: float = 0.9,
                 poison_ratio: float = 0.1,
                 additional_info: dict = {}
                 ):
        FedAvgClient.__init__(self,
                              client_id,
                              model,
                              dataloader,
                              device=device,
                              logger=logger,
                              local_epoch=local_epoch,
                              local_lr=local_lr,
                              optimizer=optimizer,
                              loss_func=loss_func,
                              sgd_momentum=sgd_momentum,
                              additional_info=additional_info)
        RandomLabelFlippingAttacker.__init__(self,
                                             poison_ratio=poison_ratio)


class FedAvgBackdoorClient(BackdoorAttacker, FedAvgClient):
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 dataloader: Data.DataLoader,
                 device: str = 'cpu',
                 from_label: int = None,
                 to_label: int = 0,
                 trigger: np.ndarray = np.ones([3, 3]),
                 start_x: int = 0,
                 start_y: int = 0,
                 logger: loguru._Logger = loguru.logger,
                 local_epoch: int = 10,
                 local_lr: float = 0.001,
                 optimizer: str = 'default',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 sgd_momentum: float = 0.9,
                 poison_ratio: float = 0.1,
                 additional_info: dict = {}
                 ):
        FedAvgClient.__init__(self,
                              client_id,
                              model,
                              dataloader,
                              device=device,
                              logger=logger,
                              local_epoch=local_epoch,
                              local_lr=local_lr,
                              optimizer=optimizer,
                              loss_func=loss_func,
                              sgd_momentum=sgd_momentum,
                              additional_info=additional_info)
        BackdoorAttacker.__init__(self,
                                  from_label=from_label,
                                  to_label=to_label,
                                  trigger=trigger,
                                  start_x=start_x,
                                  start_y=start_y,
                                  poison_ratio=poison_ratio)


class FedAvgScalingBackdoorClient(ScalingBackdoorAttacker, FedAvgClient):
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 dataloader: Data.DataLoader,
                 device: str = 'cpu',
                 scale_ratio: float = 1.0,
                 from_label: int = None,
                 to_label: int = 0,
                 trigger: np.ndarray = np.ones([3, 3]),
                 start_x: int = 0,
                 start_y: int = 0,
                 logger: loguru._Logger = loguru.logger,
                 local_epoch: int = 10,
                 local_lr: float = 0.001,
                 optimizer: str = 'default',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 sgd_momentum: float = 0.9,
                 poison_ratio: float = 0.1,
                 additional_info: dict = {}
                 ):
        FedAvgClient.__init__(self,
                              client_id,
                              model,
                              dataloader,
                              device=device,
                              logger=logger,
                              local_epoch=local_epoch,
                              local_lr=local_lr,
                              optimizer=optimizer,
                              loss_func=loss_func,
                              sgd_momentum=sgd_momentum,
                              additional_info=additional_info)
        ScalingBackdoorAttacker.__init__(self,
                                         from_label=from_label,
                                         to_label=to_label,
                                         trigger=trigger,
                                         start_x=start_x,
                                         start_y=start_y,
                                         poison_ratio=poison_ratio,
                                         scale_ratio=scale_ratio)


class FedAvgGaussianClient(GaussianNoiseAttacker, FedAvgClient):
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 dataloader: Data.DataLoader,
                 device: str = 'cpu',
                 logger: loguru._Logger = loguru.logger,
                 local_epoch: int = 10,
                 local_lr: float = 0.001,
                 optimizer: str = 'default',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 sgd_momentum: float = 0.9,
                 additional_info: dict = {}
                 ):
        FedAvgClient.__init__(self,
                              client_id,
                              model,
                              dataloader,
                              device=device,
                              logger=logger,
                              local_epoch=local_epoch,
                              local_lr=local_lr,
                              optimizer=optimizer,
                              loss_func=loss_func,
                              sgd_momentum=sgd_momentum,
                              additional_info=additional_info)
        GaussianNoiseAttacker.__init__(self)


class FedAvgLITClient(LITAttacker, FedAvgClient):
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 dataloader: Data.DataLoader,
                 device: str = 'cpu',
                 logger: loguru._Logger = loguru.logger,
                 local_epoch: int = 10,
                 local_lr: float = 0.001,
                 optimizer: str = 'default',
                 loss_func=torch.nn.CrossEntropyLoss(),
                 sgd_momentum: float = 0.9,
                 additional_info: dict = {},
                 from_label: int = None,
                 to_label: int = 0,
                 trigger: np.ndarray = np.ones([3, 3]),
                 start_x: int = 0,
                 start_y: int = 0,
                 poison_ratio: float = 0.1
                 ):
        FedAvgClient.__init__(self,
                              client_id,
                              model,
                              dataloader,
                              device=device,
                              logger=logger,
                              local_epoch=local_epoch,
                              local_lr=local_lr,
                              optimizer=optimizer,
                              loss_func=loss_func,
                              sgd_momentum=sgd_momentum,
                              additional_info=additional_info)
        BackdoorAttacker.__init__(self,
                                  from_label=from_label,
                                  to_label=to_label,
                                  trigger=trigger,
                                  start_x=start_x,
                                  start_y=start_y,
                                  poison_ratio=poison_ratio)


