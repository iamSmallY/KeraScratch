"""神经网络层模块。"""
from abc import ABCMeta, abstractmethod
import numpy as np

from PyTorScratch.tensor import Tensor


class Operator(metaclass=ABCMeta):
    """神经网络算子抽象类。

    对于一个神经网络中的某一算子而言，仅需要一个前向传播，以及一个反向传播方法。
    """

    def args(self):
        """获取算子全部参数方法。

        获取当前算子所含的全部张量即可。
        """
        res = []
        for name in vars(self).keys():
            if isinstance(getattr(self, name), Tensor):
                res.append(getattr(self, name))
        return res

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """计算前向传播方法。

        将张量 x 通过该算子进行前向传播，计算其结果，并返回结果张量。

        Args:
            x: 用于计算前向传播的张量

        Returns:
            计算得到的结果张量。
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def backward(**kwargs: Tensor) -> None:
        """计算反向传播方法。

        根据算子计算得到的张量 y 的值，更新张量 x 的梯度值。\n
        该函数应用于在前向传播时被放入张量的 bp_cache 中。

        Args:
            kwargs: 反向传播所用参数
        """
        raise NotImplementedError()


class LinearOperator(Operator):
    """全连接层。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """初始化全连接层方法。

        全连接算子计算方式为 y = wx + b.\n
        而对于张量 x 和 y，它们的维度可以是不同的，所以要利用这两个张量的维度来决定 w 和 b 的形状。

        Args:
            in_dim: 将被输入的张量的维度
            out_dim: 需要张量的维度
        """
        self.__out_dim = out_dim
        self.__w = Tensor(np.random.randn(in_dim, out_dim) * 0.1)
        self.__b = Tensor(np.random.randn(1, out_dim) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        y = Tensor(np.dot(x.value, self.__w.value) + self.__b.value)
        y.append_bp_cache(self.backward, {'x': x, 'w': self.__w, 'b': self.__b})
        return y

    @staticmethod
    def backward(**kwargs: Tensor) -> None:
        y = kwargs['y']
        x = kwargs['x']
        w = kwargs['w']
        b = kwargs['b']
        x.grad += np.dot(y.grad, w.value.T)
        w.grad += np.dot(x.value.T, y.grad)
        b.grad += y.grad