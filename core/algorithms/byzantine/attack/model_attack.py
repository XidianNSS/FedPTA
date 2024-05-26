from fedsimu.core.algorithms.base.base_attacker import BaseAttacker
from copy import deepcopy
from collections import OrderedDict
import torch


class ModelAttacker(BaseAttacker):
    def __init__(self):
        BaseAttacker.__init__(self)


class GaussianNoiseAttacker(ModelAttacker):
    def __init__(self):
        ModelAttacker.__init__(self)
        self.local_records['attacker_type'] = 'fedsimu.core.algorithms.byzantine.attack.model_attack.' \
                                              'GaussianNoiseAttacker'

    def attack_before_train(self, global_info: dict):
        pass

    def attack_after_train(self, global_info: dict):
        self.model = self.model.to('cpu')

        grad_dict = self.model.state_dict()
        grad_dict_gaussian = OrderedDict()
        for k, v in grad_dict.items():
            noise = torch.randn(v.shape).to('cpu')
            # var_GS = var + noise * torch.std(var) * 2
            a = torch.mean(v)
            b = torch.std(v)
            v_gaussian = a + noise * b * 2
            grad_dict_gaussian[k] = v_gaussian

        self.model.load_state_dict(grad_dict_gaussian, strict=True)
