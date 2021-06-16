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
        pass

    @staticmethod
    @abstractmethod
    def backward(x: Tensor, target: Tensor) -> None:
        """计算损失函数反向传播值。

        Args:
            x: 当前层张量
            target: 损失函数结果张量
        """
        pass


class CrossEntropyLoss(Loss):
    """交叉熵损失函数类。

    利用输出层的 softmax 结果计算交叉熵损失函数值。
    """
    @staticmethod
    def softmax(x: Tensor) -> Tensor:
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
        s = CrossEntropyLoss.softmax(x).value
        l = -np.log(s[0][target.value[0]])
        y = Tensor(np.array([l]))
        y.append_bp_cache(CrossEntropyLoss.backward, {'x': x, 'target': target})
        return y

    @staticmethod
    def backward(x: Tensor, target: Tensor) -> None:
        x.grad = CrossEntropyLoss.softmax(x)
        x.grad[0][target.value[0]] -= 1