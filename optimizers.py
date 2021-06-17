"""优化器模块"""
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List

from tensor import Tensor


class Optimizer(metaclass=ABCMeta):
    """优化器抽象类。"""

    @abstractmethod
    def step(self):
        """更新模型的梯度值方法。

        在反向传播完成后更新模型中各张量的值。
        """
        pass


class SGD(Optimizer):
    """带动量的随机梯度下降优化器。"""

    def __init__(self, args: List[Tensor], learning_rate: float = 1e-3, momentum: float = 0.5):
        self.__args = args
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__v = [np.zeros(arg.value.shape) for arg in args]

    def step(self):
        for i in range(len(self.__args)):
            self.__v[i] = self.__momentum * self.__v[i] - self.__learning_rate * self.__args[i].grad
            self.__args[i].value += self.__v[i]
