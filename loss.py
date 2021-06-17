"""损失函数模块。"""
from abc import ABCMeta, abstractmethod
import numpy as np

from tensor import Tensor


class Loss(metaclass=ABCMeta):
    """损失函数抽象类。"""

    @staticmethod
    @abstractmethod
    def calculate(x: Tensor, target: Tensor) -> Tensor:
        """计算张量 x 与 target 之间的损失值 f(x, target)

        Args:
            x: 神经网络输出张量
            target: 目标结果张量

        Returns:
            f(x, target).
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def backward(**kwargs) -> None:
        """计算损失函数反向传播值。

        Args:
            kwargs: 反向传播函数所用参数
        """
        raise NotImplementedError()


class CrossEntropyLoss(Loss):
    """交叉熵损失函数类。

    内部整合了 softmax 函数，便于计算，在使用模型时最后一层无需 softmax 函数。
    """

    @staticmethod
    def softmax(x: Tensor) -> np.ndarray:
        """计算 softmax 函数方法。

        Args:
            x: 输入张量

        Returns:
            softmax(x)
        """
        e = np.exp(x.value)
        return e / np.sum(e)

    @staticmethod
    def calculate(x: Tensor, target: Tensor) -> Tensor:
        s = CrossEntropyLoss.softmax(x)
        loss = -np.log(s[0][target.value[0]])
        y = Tensor(np.array([loss]))
        y.append_bp_cache(CrossEntropyLoss.backward, {'x': x, 'target': target})
        return y

    @staticmethod
    def backward(**kwargs: Tensor) -> None:
        x = kwargs['x']
        target = kwargs['target']
        x.grad = CrossEntropyLoss.softmax(x)
        x.grad[0][target.value[0]] -= 1
