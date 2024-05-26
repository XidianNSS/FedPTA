from fedsimu.core.algorithms.base.base_attacker import BaseAttacker
from copy import deepcopy
from fedsimu.reporter.utils import get_x_y_from_dataloader
import numpy as np
import random
import math
from fedsimu.benchmark.partition import MyDataset
import torch.utils.data as Data
from fedsimu.core.algorithms.utils import replace_data
from fedsimu.core.algorithms.utils import serialize_model, deserialize_model


class DataPoisoningAttacker(BaseAttacker):
    """
    This class implements a base class for data poisoning attackers.

    """
    def __init__(self, poison_ratio: float = 0.1):
        """

        :param poison_ratio: float, make sure that (Data_poison / Data_total == poison_ratio)
        """
        BaseAttacker.__init__(self)

        self.origin_dataloader = deepcopy(self.dataloader)  # origin dataloader is for backup

        x_numpy, y_numpy = get_x_y_from_dataloader(self.origin_dataloader)
        self.origin_x_numpy = x_numpy  # local dataset, in numpy
        self.origin_y_numpy = y_numpy

        assert 0 <= poison_ratio <= 1  # value check
        self.poison_ratio = poison_ratio
        self.local_records['poison_ratio'] = self.poison_ratio
        self.local_records['dataloader'] = self.origin_dataloader


class RandomLabelFlippingAttacker(DataPoisoningAttacker):

    def __init__(self, poison_ratio: float = 0.1):
        DataPoisoningAttacker.__init__(self, poison_ratio=poison_ratio)
        self.local_records['attacker_type'] = 'fedsimu.core.algorithms.byzantine.attack.data_poisoning.' \
                                              'RandomLabelFlippingAttacker'

    def attack_before_train(self, global_info: dict):
        unique_labels = np.unique(self.origin_y_numpy)
        total_samples = len(self.origin_x_numpy)

        # poisoning dataset
        self.logger.info('Attacker is flipping labels.')
        poisoned_samples_num = math.floor(total_samples * self.poison_ratio)
        poisoned_samples_idx = random.sample(range(total_samples), k=poisoned_samples_num)
        poisoned_y = deepcopy(self.origin_y_numpy)

        for idx in poisoned_samples_idx:
            origin_label = poisoned_y[idx]

            random_label = origin_label
            max_try_times = 20
            try_times = 0
            while random_label == origin_label:
                try_times += 1
                random_label = random.choice(unique_labels)
                if try_times > max_try_times:
                    self.logger.warning('Trying to flip a sample to random label but fails.')
                    break

            poisoned_y[idx] = random_label

        poisoned_dataset = MyDataset(self.origin_x_numpy, poisoned_y)
        self.dataloader = Data.DataLoader(poisoned_dataset, batch_size=self.origin_dataloader.batch_size)
        self.logger.info(f'Attacker has flipped labels with poisoned num = {poisoned_samples_num}.')

    def attack_after_train(self, global_info: dict):
        pass


class BackdoorAttacker(DataPoisoningAttacker):

    def __init__(self,
                 poison_ratio: float = 0.1,
                 from_label: int = None,
                 to_label: int = 0,
                 trigger: np.ndarray = np.ones([3, 3]),
                 start_x: int = 0,
                 start_y: int = 0
                 ):
        """
        :param trigger  See replace_data
        :param start_x
        :param start_y
        :param poison ratio: Poisoned dataset amount
        :param from_label: Add trigger to certain label data
        :param to_label: Backdoor data will be relabeled to it.
        """
        DataPoisoningAttacker.__init__(self, poison_ratio=poison_ratio)
        self.local_records['attacker_type'] = 'fedsimu.core.algorithms.byzantine.attack.data_poisoning.' \
                                              'BackdoorAttacker'
        self.trigger = trigger
        self.start_x = start_x
        self.start_y = start_y
        self.from_label = from_label
        self.to_label = to_label
        self.local_records.update({
            'backdoor': True,
            'backdoor_trigger': self.trigger,
            'backdoor_start_x': self.start_x,
            'backdoor_start_y': self.start_y,
            'backdoor_from_label': self.from_label,
            'backdoor_to_label': self.to_label,
        })

    def attack_before_train(self, global_info: dict):
        self.logger.info('Attacker is inserting backdoors.')
        total_samples = len(self.origin_x_numpy)

        poisoned_samples_num = math.floor(total_samples * self.poison_ratio)
        poisoned_x = deepcopy(self.origin_x_numpy)
        poisoned_y = deepcopy(self.origin_y_numpy)

        # get idx of data samples to backdoor
        poisoned_samples_idx = None
        if self.from_label is None:
            # random sample
            poisoned_samples_idx = random.sample(range(total_samples), k=poisoned_samples_num)
        else:
            # sample label = from_label
            poisoned_samples_idx = np.where(self.origin_y_numpy == self.from_label)[0]
            if len(poisoned_samples_idx > poisoned_samples_num):
                poisoned_samples_idx = poisoned_samples_idx[0: poisoned_samples_num]

        poisoned_y[poisoned_samples_idx] = self.to_label
        poisoned_samples_num = len(poisoned_samples_idx)

        if len(self.origin_x_numpy.shape) == 3:
            # 1 tunnel
            for idx in poisoned_samples_idx:
                replace_data(poisoned_x[idx], self.trigger, self.start_x, self.start_y)

        elif len(self.origin_x_numpy.shape) == 4:
            # multi tunnels
            for tunnel in range(self.origin_x_numpy.shape[1]):
                for idx in poisoned_samples_idx:
                    replace_data(poisoned_x[idx][tunnel], self.trigger, self.start_x, self.start_y)
        else:
            self.logger.error('Unknown picture shape. X of picture datasets should have '
                              'len(self.origin_x_numpy.shape) == 3 or len(self.origin_x_numpy.shape) == 4, but'
                              f'got {len(self.origin_x_numpy.shape)} instead. shape = {self.origin_x_numpy.shape}')
            raise TypeError

        poisoned_dataset = MyDataset(poisoned_x, poisoned_y)
        self.dataloader = Data.DataLoader(poisoned_dataset, batch_size=self.origin_dataloader.batch_size)
        self.logger.info(f'Attacker has inserted backdoors with poisoned num = {poisoned_samples_num}.')

    def attack_after_train(self, global_info: dict):
        self.uplink_package.update({
            'backdoor': True,
            'backdoor_trigger': self.trigger,
            'backdoor_start_x': self.start_x,
            'backdoor_start_y': self.start_y,
            'backdoor_from_label': self.from_label,
            'backdoor_to_label': self.to_label,
        })
        self.dataloader = deepcopy(self.origin_dataloader)  # resume normal dataloader


class ScalingBackdoorAttacker(BackdoorAttacker):
    def __init__(self,
                 poison_ratio: float = 0.1,
                 from_label: int = None,
                 to_label: int = 0,
                 trigger: np.ndarray = np.ones([3, 3]),
                 start_x: int = 0,
                 start_y: int = 0,
                 scale_ratio: float = 1.0
                 ):
        """

        Args:
            poison_ratio:
            from_label:
            to_label:
            trigger:
            start_x:
            start_y:
            scale_ratio: local param will be rescaled before uploaded to server.
                         param = (ori_param - global_param) * scale_ratio + global_param
        """

        BackdoorAttacker.__init__(self,
                                  poison_ratio=poison_ratio,
                                  from_label=from_label,
                                  to_label=to_label,
                                  trigger=trigger,
                                  start_x=start_x,
                                  start_y=start_y
                                  )
        self.local_records['attacker_type'] = 'fedsimu.core.algorithms.byzantine.attack.data_poisoning.' \
                                              'ScalingBackdoorAttacker'
        self.scale_ratio = scale_ratio
        self.local_records.update({
            'scale_ratio': self.scale_ratio
        })

    def attack_after_train(self, global_info: dict):
        if 'avg_parameter' not in global_info.keys():
            self.logger.error("'avg_parameter' not found in global_info. "
                              "ScalingBackdoorAttacker does not support logit updates.")
            raise TypeError
        global_param = global_info['avg_parameter']
        delta_local_param = serialize_model(self.model) - global_param
        attacked_param = delta_local_param * self.scale_ratio + global_param
        deserialize_model(self.model, attacked_param)





