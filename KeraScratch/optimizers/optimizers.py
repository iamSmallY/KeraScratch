"""优化器模块"""
from abc import ABCMeta, abstractmethod
from typing import List

from KeraScratch.tensor import Tensor


class Optimizer(metaclass=ABCMeta):
    """优化器抽象类。"""

    @abstractmethod
    def step(self, args: List[Tensor]):
        """更新模型的梯度值方法。

        在反向传播完成后更新模型中各张量的值。

        Args:
            args: 模型中各个参数张量的列表，用于更新其梯度
        """
        raise NotImplementedError()
