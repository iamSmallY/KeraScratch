import numpy as np
from typing import List

from .optimizers import Optimizer
from KeraScratch.tensor import Tensor


class SGD(Optimizer):
    """带动量的随机梯度下降优化器。"""

    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.5):
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__v = None

    def step(self, args: List[Tensor]):
        if self.__v is None:
            self.__v = [np.zeros(arg.value.shape) for arg in args]

        for i in range(len(args)):
            self.__v[i] = self.__momentum * self.__v[i] - self.__learning_rate * args[i].grad
            args[i].value += self.__v[i]
