from abc import abstractmethod
import numpy as np

from PyTorScratch.operators.operators import Operator
from PyTorScratch.tensor import Tensor


class ActivationOperator(Operator):
    """激活函数算子抽象类

    扩展了 Operator 类，增加了激活函数算子所需要用到的计算激活函数的方法。
    """

    @staticmethod
    @abstractmethod
    def function(x: Tensor) -> Tensor:
        """计算张量经过激活函数后的值"""
        raise NotImplementedError()


class ReLUOperator(ActivationOperator):
    """ReLU 算子。"""

    @staticmethod
    def function(x: Tensor) -> Tensor:
        return Tensor((x.value > 0).astype(np.float32) * x.value)

    def forward(self, x: Tensor) -> Tensor:
        y = self.function(x)
        y.append_bp_cache(self.backward, {'x': x})
        return y

    @staticmethod
    def backward(**kwargs: Tensor) -> None:
        y = kwargs['y']
        x = kwargs['x']
        x.grad += y.grad * (y.value > 0).astype(np.float32)


class SigmoidOperator(ActivationOperator):
    """Sigmoid 算子。"""

    @staticmethod
    def function(x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x.value)))

    def forward(self, x: Tensor) -> Tensor:
        y = self.function(x)
        y.append_bp_cache(self.backward, {'x': x})
        return y

    @staticmethod
    def backward(**kwargs: Tensor) -> None:
        y = kwargs['y']
        x = kwargs['x']
        x.grad += y.grad * SigmoidOperator.function(x).value * (1 - SigmoidOperator.function(x).value)
