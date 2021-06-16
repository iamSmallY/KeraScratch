"""神经网络层模块。"""
from abc import ABCMeta, abstractmethod
import numpy as np

from tensor import Tensor


class Layer(metaclass=ABCMeta):
    """神经网络层抽象类。

    对于一个神经网络中的某一层而言，仅需要一个前向传播，以及一个反向传播方法。
    """
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """计算前向传播方法。

        将张量 x 通过该层进行前向传播，计算其结果，并返回结果张量。

        Args:
            x: 用于计算前向传播的张量

        Returns:
            计算得到的结果张量。
        """
        pass

    @staticmethod
    @abstractmethod
    def backward(x: Tensor, y: Tensor) -> None:
        """计算反向传播方法。

        根据后一层张量 y 的值，更新张量 x 的梯度值。\n
        该函数应用于在前向传播时被放入张量的 bp_cache 中。

        Args:
            x: 待梯度更新的张量
            y: 后一层与 x 对应的张量
        """
        pass


class LinearLayer(Layer):
    """全连接层。"""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        """初始化全连接层方法。

        全连接层计算方式为 y = wx + b.\n
        而对于张量 x 和 y，它们的维度可以是不同的，所以要利用这两个张量的维度来决定 w 和 b 的形状。

        Args:
            in_dim: x 张量的维度
            out_dim: y 张量的维度
        """
        self.__out_dim = out_dim
        self.__w = Tensor(np.random.randn(in_dim, out_dim) * 0.1)
        self.__b = Tensor(np.random.randn(1, out_dim) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        y = Tensor(np.dot(x.value, self.__w.value) + self.__b.value)
        y.append_bp_cache(self.backward, {'x': x, 'w': self.__w, 'b': self.__b})
        return y

    def backward(self, x: Tensor, y: Tensor) -> None:
        x.grad += np.dot(y.grad, self.__w.value.T)
        self.__w.grad += np.dot(x.value.T, y.grad)
        self.__b.grad += y.grad


class ActivationLayer(Layer):
    """激活函数层抽象类

    扩展了 Layer 类，增加了激活函数层所需要用到的计算激活函数的方法。
    """
    @staticmethod
    @abstractmethod
    def function(x: Tensor) -> Tensor:
        """计算张量经过激活函数后的值"""
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def backward(self, x: Tensor, y: Tensor) -> None:
        pass


class ReLULayer(ActivationLayer):
    """ReLU 层。"""
    @staticmethod
    def function(x: Tensor) -> Tensor:
        return Tensor((x.value > 0).astype(np.float32) * x.value)

    def forward(self, x: Tensor) -> Tensor:
        y = self.function(x)
        y.append_bp_cache(self.backward, {'x': x})
        return y

    def backward(self, x: Tensor, y: Tensor) -> None:
        x.grad += y.grad * (y.value > 0).astype(np.float32)


class SigmoidLayer(ActivationLayer):
    """Sigmoid 层。"""
    @staticmethod
    def function(x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x.value)))

    def forward(self, x: Tensor) -> Tensor:
        y = self.function(x)
        y.append_bp_cache(self.backward, {'x': x})
        return y

    def backward(self, x: Tensor, y: Tensor) -> None:
        x.grad += y.grad * self.function(x).value * (1 - self.function(x).value)
