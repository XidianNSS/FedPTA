import torch
from abc import abstractmethod
import torch.utils.data as Data
import numpy as np


class RecommendOptim(object):
    @abstractmethod
    def recommended_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError



