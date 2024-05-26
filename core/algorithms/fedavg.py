import torch
from fedsimu.core.algorithms.base.base_client import BaseClient
from fedsimu.core.algorithms.base.base_server import BaseTestServer
from fedsimu.core.algorithms.base.base_sampler import BaseClientSampler
import torch.utils.data as Data
import loguru
from fedsimu.core.algorithms.utils import serialize_model, deserialize_model
from typing import List, Union
from copy import deepcopy
import numpy as np


class FedAvgClient(BaseClient):
    """

    Transport:
        Svr -> Cli:
            'avg_parameter': np.ndarray
            'round': int

        Cli -> Svr:
            'client_id': int
            'parameter': np.ndarray
            'round': int

    """

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
        BaseClient.__init__(self,
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
                            additional_info=additional_info
                            )

    def before_train(self, global_info: dict, record_info: bool = True):
        self.model = self.model.to('cpu')
        deserialize_model(self.model, global_info['avg_parameter'])

    def train(self, global_info: dict, record_info: bool = True):
        local_loss_this_round = []
        local_acc_this_round = []
        self.model = self.model.to(self.device)

        for i in range(self.local_epoch):
            correct = 0
            total = 0  # For acc calculate

            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x, y = x.float(), y.float()
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                log_probs = self.model(x)
                loss = self.loss_func(log_probs, y.long())
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                total += len(y_pred)

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss = sum(batch_loss) / len(batch_loss)
            epoch_acc = correct / total

            self.logger.debug(
                "[Round {}]Training on client {}, local epoch {} / {}, epoch loss {:.2f}, epoch acc {:.2f}%"
                .format(global_info['round'] + 1,
                        self.client_id, i + 1,
                        self.local_epoch,
                        epoch_loss,
                        epoch_acc * 100))

            local_loss_this_round.append(epoch_loss)
            local_acc_this_round.append(epoch_acc)

        self.logger.info("[Round {}]Trained on client {}, average train loss {:.2f}, average train acc {:.2f}%"
                         .format(global_info['round'] + 1,
                                 self.client_id,
                                 sum(local_loss_this_round) / self.local_epoch,
                                 sum(local_acc_this_round) * 100 / self.local_epoch))
        if record_info:
            self.local_records['train_loss'].append(local_loss_this_round)
            self.local_records['train_acc'].append(local_acc_this_round)
        self.model = self.model.to('cpu')
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

    def after_train(self, global_info: dict, record_info: bool = True):

        self.uplink_package['parameter'] = serialize_model(self.model)
        self.uplink_package['dataloader_size'] = self.dataloader_size


class FedAvgServer(BaseTestServer):
    """

        Transport:
            Svr -> Cli:
                'avg_parameter': np.ndarray
                'round': int

            Cli -> Svr:
                'client_id': int
                'parameter': np.ndarray
                'round': int

    """

    def __init__(self,
                 test_model: torch.nn.Module,
                 test_dataloader: Data.DataLoader,
                 device: str = 'cpu',
                 sampler: BaseClientSampler = None,
                 logger: loguru._Logger = loguru.logger,
                 backdoor_experiment: bool = False,
                 backdoor_from_label: int = None,
                 backdoor_to_label: int = 0,
                 backdoor_trigger: np.ndarray = np.ones([3, 3]),
                 backdoor_start_x: int = 0,
                 backdoor_start_y: int = 0
                 ):
        super().__init__(test_model,
                         test_dataloader,
                         device,
                         sampler,
                         logger,
                         backdoor_experiment=backdoor_experiment,
                         backdoor_from_label = backdoor_from_label,
                         backdoor_to_label = backdoor_to_label,
                         backdoor_trigger = backdoor_trigger,
                         backdoor_start_x = backdoor_start_x,
                         backdoor_start_y = backdoor_start_y
                         )

    def before_test(self, all_client_info: List[dict]):
        avg_parameters = None
        client_total_dataloader_size = 0

        for client_info in all_client_info:
            if not isinstance(client_info, dict):
                self.logger.error(f'Client info is not a dict! Check if you have any type error. Client info '
                                  f'received: {client_info}')
                raise TypeError
            client_dataloader_size = client_info['dataloader_size']
            client_total_dataloader_size += client_dataloader_size

            client_parameter = client_info['parameter']

            if avg_parameters is None:
                avg_parameters = deepcopy(client_parameter) * client_dataloader_size
            else:
                avg_parameters += client_parameter * client_dataloader_size

        avg_parameters = avg_parameters / client_total_dataloader_size
        self.uplink_package['avg_parameter'] = deepcopy(avg_parameters)

        deserialize_model(self.test_model, avg_parameters)
        # self.logger.debug(f'Server send Avg parameters (3 parameters) {avg_parameters[0:3]}')


