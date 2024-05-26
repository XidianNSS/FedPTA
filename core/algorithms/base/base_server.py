import torch
import loguru
import torch.utils.data as Data
from copy import deepcopy
from abc import abstractmethod, ABC
from typing import List
from .base_sampler import BaseClientSampler, AllClientSampler
from fedsimu.core.algorithms.base.recordable import Recordable
import numpy as np
from fedsimu.core.algorithms.utils import replace_data
from fedsimu.reporter.utils import get_x_y_from_dataloader
from fedsimu.benchmark.partition import MyDataset

class BaseServer(Recordable):
    def __init__(self,
                 sampler: BaseClientSampler = None,
                 logger: loguru._Logger = loguru.logger):
        self.round = 0
        '''
            Round 0: svr -> cli init.
            Round 1: cli train -> svr, svr agg.
            Round 2: cli train -> svr, svr agg.
            ...
        
        '''

        self.uplink_package = {}
        self.logger = logger

        if sampler is None:
            self.sampler = AllClientSampler(logger=logger)
        else:
            self.sampler = sampler

        self.records = {
            'sampler': str(self.sampler.__class__),
            'class': str(self.__class__)
        }

    def local_process(self, all_client_info: List[dict]) -> dict:
        self.round += 1
        self.uplink_package = {'round': self.round}
        all_client_info = self.sampler.sample_client(all_client_info)  # Sample client here.
        self.before_test(all_client_info)
        self.test(all_client_info)
        self.after_test(all_client_info)

        return deepcopy(self.uplink_package)

    def test(self, all_client_info: List[dict]):
        """
        Perform test on test_dataset with test_model. Record info will be in self.records

        :param all_client_info: Info from all clients. An iterable object.
        """
        pass

    def before_test(self, all_client_info: List[dict]):
        """
        Aggregate info from clients. You can define other process in this method

        :param all_client_info: Info from all clients. An iterable object.
        """

        pass

    def after_test(self, all_client_info: List[dict]):
        """
        :param all_client_info: Info from all clients. An iterable object.
        """
        pass

    def get_record_abstract(self) -> dict:
        self.records.update({
            'rounds': self.round
        })
        self.records.update(self.sampler.get_record_abstract())
        return deepcopy(self.records)


class BaseTestServer(BaseServer, ABC):
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

        super().__init__(sampler=sampler, logger=logger)
        self.test_model = deepcopy(test_model)
        self.test_dataloader = deepcopy(test_dataloader)
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.records.update({
            'test_acc': [],
            'test_loss': [],
            'backdoor_asr': [],
            'test_model': str(self.test_model.__class__),
            'test_dataloader': self.test_dataloader,
            'test_loss_func': str(self.loss_func.__class__)
        })
        self.backdoor_experiment = backdoor_experiment
        self.backdoor_trigger = backdoor_trigger
        self.backdoor_start_x = backdoor_start_x
        self.backdoor_start_y = backdoor_start_y
        self.backdoor_from_label = backdoor_from_label
        self.backdoor_to_label = backdoor_to_label

    def test(self, all_client_info: List[dict]):
        self.test_model = self.test_model.to(self.device)

        correct_nums = 0
        total_nums = 0
        batch_loss = []
        for idx, (x, y) in enumerate(self.test_dataloader):
            x, y = x.float(), y.float()
            x, y = x.to(self.device), y.to(self.device)

            log_probs = self.test_model(x)
            loss = self.loss_func(log_probs, y.long())
            predict = log_probs.data.max(1, keepdim=True)[1]

            correct_nums += predict.eq(y.data.view_as(predict)).long().cpu().sum()
            total_nums += len(predict)

            batch_loss.append(loss.item())

        test_loss = sum(batch_loss) / len(batch_loss)
        test_acc = 1.0 * correct_nums / total_nums
        self.records['test_acc'].append(test_acc)
        self.records['test_loss'].append(test_loss)
        self.logger.info("[Round {}]Testing on server, loss {:.2f}, acc {:.2f}%"
                         .format(self.round, test_loss, test_acc * 100))
        self.test_model = self.test_model.to('cpu')
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

        # check if this is a backdoor experiment.
        # if self.round == 1:
        #     for client_info in all_client_info:  # unzip client info
        #         if 'backdoor' in client_info.keys():
        #             self.backdoor_experiment = True
        #             self.backdoor_trigger = client_info['backdoor_trigger']
        #             self.backdoor_start_x = client_info['backdoor_start_x']
        #             self.backdoor_start_y = client_info['backdoor_start_y']
        #             self.backdoor_from_label = client_info['backdoor_from_label']
        #             self.backdoor_to_label = client_info['backdoor_to_label']

        if self.backdoor_experiment:
            self.records['backdoor'] = True
            assert self.backdoor_from_label != self.backdoor_to_label  # from label and to label cannot be same
            origin_x, origin_y = get_x_y_from_dataloader(self.test_dataloader)

            if self.backdoor_from_label is None:
                to_test_idx = np.where(origin_y != self.backdoor_from_label)
            else:
                to_test_idx = np.where(origin_y == self.backdoor_from_label)

            origin_x = origin_x[to_test_idx]
            origin_y = origin_y[to_test_idx]

            if len(origin_x.shape) == 3:
                for idx in range(len(origin_x)):
                    replace_data(origin_x[idx], self.backdoor_trigger, self.backdoor_start_x, self.backdoor_start_y)
            elif len(origin_x.shape) == 4:
                for tunnel in range(origin_x.shape[1]):
                    for idx in range(len(origin_x)):
                        replace_data(origin_x[idx][tunnel],
                                     self.backdoor_trigger,
                                     self.backdoor_start_x,
                                     self.backdoor_start_y)

            for idx in range(len(origin_y)):
                origin_y[idx] = self.backdoor_to_label

            backdoor_dataset = MyDataset(origin_x, origin_y)
            backdoor_dataloader = Data.DataLoader(backdoor_dataset, batch_size=32)

            self.test_model = self.test_model.to(self.device)
            correct_nums = 0
            total_nums = 0

            for idx, (x, y) in enumerate(backdoor_dataloader):
                x, y = x.float(), y.float()
                x, y = x.to(self.device), y.to(self.device)

                log_probs = self.test_model(x)
                predict = log_probs.data.max(1, keepdim=True)[1]
                correct_nums += predict.eq(y.data.view_as(predict)).long().cpu().sum()
                total_nums += len(predict)

            asr = 1.0 * correct_nums / total_nums
            self.records['backdoor_asr'].append(asr)
            self.logger.info("[Round {}]Testing ASR on server, ASR {:.2f}%"
                             .format(self.round, asr * 100))
            self.test_model = self.test_model.to('cpu')
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

    def get_record_abstract(self) -> dict:
        record = super().get_record_abstract()
        record.update({'backdoor': self.backdoor_experiment})
        return record

