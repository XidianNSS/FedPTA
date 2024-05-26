import torch
import loguru
import torch.utils.data as Data
from copy import deepcopy
from abc import abstractmethod
from fedsimu.core.algorithms.utils import serialize_model
from fedsimu.benchmark.utils import RecommendOptim
from fedsimu.core.algorithms.base.recordable import Recordable
from fedsimu.reporter.utils import get_y_from_dataloader


class BaseClient(Recordable):
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

                 ):
        """

        :param client_id:
        :param model:
        :param dataloader:
        :param device:
        :param logger:
        :param local_epoch:
        :param local_lr: if optimizer is a torch.optim.optimizer.Optimizer, this is useless.
        :param optimizer:
             optimizer is a str:
                accepted: ['sgd', 'adam', 'default']

        :param loss_func:
        :param sgd_momentum:
        :param additional_info: extra info from global

        """
        self.logger = logger
        self.client_id = client_id
        self.model = deepcopy(model)
        self.device = device
        self.dataloader = deepcopy(dataloader)
        self.local_epoch = local_epoch  # local epoch
        self.local_lr = local_lr  # local learning rate
        self.sgd_momentum = sgd_momentum
        self.loss_func = loss_func  # Default loss function. You can redefine it in before_train
        self.uplink_package = {'client_id': self.client_id}


        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.local_lr,
                                             momentum=sgd_momentum)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.local_lr)

        elif optimizer == 'default':
            if isinstance(self.model, RecommendOptim):
                self.optimizer = self.model.recommended_optimizer()
            else:
                self.logger.warning('Tryed to use default optimizer from recommended_optimizer(), but you ' +
                                    'the model is not inherited from RecommendOptim. Using SGD instead.')
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=self.local_lr,
                                                 momentum=sgd_momentum)

        else:
            self.logger.error(f"Not supported optimizer '{optimizer}', please try to implement it.")
            raise NotImplementedError()

        y = get_y_from_dataloader(self.dataloader)
        self.dataloader_size = len(y)
        self.local_records = {
            'train_loss': [],
            'train_acc': [],
            'client_id': self.client_id,
            'dataloader': deepcopy(self.dataloader),
            'dataloader_size': self.dataloader_size,
            'model': str(self.model.__class__),
            'local_epoch': self.local_epoch,
            'optimizer': str(self.optimizer.__class__),
            'optimizer_lr': self.optimizer.defaults['lr'],
            'loss_func': str(self.loss_func.__class__),
            'class': str(self.__class__),
            'dataset_name': str(self.dataloader.dataset),
            'local_history_parameters': []
        }

        self.local_records.update(additional_info)

    def local_process(self, global_info: dict) -> dict:
        """
        Define the whole local process in the client.

        Warning! This function should not be rewritten if you want to create an attacker with it. This is because
        BaseAttacker rewrites local_process(self, global_info: dict) to add attack process in it

        :param global_info: (dict) Global info from server.
        :return: (dict) Local info.
        """

        self.logger.info("[Round {}]Start training on client {}."
                         .format(global_info['round'] + 1,
                                 self.client_id))
        self.log_client_abstract()

        self.uplink_package = {'client_id': self.client_id, 'round': global_info['round'] + 1}
        self.before_train(global_info)  # Usually load global info from server.
        self.train(global_info)  # Usually train process
        self.after_train(global_info)  # Usually upload local info to server.
        self.record_model(global_info)
        return self.uplink_package

    def start(self) -> dict:
        """
        Start FL process without global_info

        Returns:

        """
        self.logger.info("[Round {}]Start training on client {}."
                         .format(1,
                                 self.client_id))
        self.log_client_abstract()

        self.uplink_package = {'client_id': self.client_id, 'round': 1}
        self.train({'round': 0})  # Usually train process
        self.after_train({'round': 0})  # Usually upload local info to server.
        self.record_model({'round': 0})
        return self.uplink_package

    @abstractmethod
    def train(self, global_info: dict, record_info: bool = True):
        """
        Train with local utils, record info will be record in self.uplink_package

        :param global_info: (dict) Global info from server.
        :param record_info: (bool) if record info in this process


        """
        pass

    @abstractmethod
    def before_train(self, global_info: dict, record_info: bool = True):
        """

        Process before train, record info will be record in self.uplink_package
        Usually this is a process to learn from the global.

        :param global_info: (dict) Global info from server.
        :param record_info: (bool) if record info in this process
        """
        pass

    @abstractmethod
    def after_train(self, global_info: dict, record_info: bool = True):
        """

        Process after train, record info will be record in self.uplink_package

        :param global_info: (dict) Global info from server.
        :param record_info: (bool) if record info in this process
        """
        pass

    def get_record_abstract(self) -> dict:
        self.local_records['model_parameters'] = serialize_model(self.model)
        return deepcopy(self.local_records)

    def log_client_abstract(self):

        self.logger.debug(f'[Client Abstract]client_id = {self.client_id}')
        self.logger.debug(f'[Client Abstract]model = {str(self.model.__class__)}')
        self.logger.debug(f'[Client Abstract]local_epoch = {self.local_epoch}')
        self.logger.debug(f'[Client Abstract]optimizer = {str(self.optimizer.__class__)}')
        self.logger.debug(f'[Client Abstract]optimizer_lr = {self.optimizer.defaults["lr"]}')
        self.logger.debug(f'[Client Abstract]loss_func = {str(self.loss_func.__class__)}')
        self.logger.debug(f'[Client Abstract]class = {str(self.__class__)}')

    def record_model(self, global_info: dict):
        """
        This process usually records local model if you need to do some experiment with model records.
        If you do not need to record ( which usually take large amount of memory ), ignore it.

        :param global_info:
        :return:
        """
        pass



