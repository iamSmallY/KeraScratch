"""神经网络层模块。"""
from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Optional

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


class Conv2DOperator(Operator):
    """2D 卷积算子。"""
    def __init__(self, out_channel: int, kernel_size: int, strides: int = 1, padding: str = 'valid') -> None:
        """初始化卷积层方法。

        Args:
            out_channel: 输出通道数
            kernel_size: 卷积核大小
            strides: 卷积步长
            padding: 卷积填充方案：
                1. valid: 不填充 (default)
                2. same: 边界填充 kernel_size / 2 大小
                3. full: 边界填充 kernel_size - 1 大小
                填充模式均为常量填充，填充数字 0.
        """
        self.__out_channel = out_channel
        self.__kernel_size = kernel_size
        self.__strides = strides
        self.__padding = padding

        self.__filter: Optional[Tensor] = None
        self.__bias: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.__filter is None:
            filter_shape = x.value.shape[-1], self.__kernel_size, self.__kernel_size, self.__out_channel
            self.__filter = Tensor(np.random.randn(*filter_shape) *
                                   (2 / reduce(lambda _x, _y: _x * _y, filter_shape[1:]) ** 0.5))
            self.__bias = Tensor(np.zeros(self.__out_channel))

        kernel_height, kernel_width, channel, out_channel = self.__filter.value.shape
        x_split = self.__split_by_strides(x.value, kernel_height, kernel_width, self.__strides)
        x_split = x_split.astype('float64') / 255
        out = np.tensordot(x_split, self.__filter.value, axes=[(3, 4, 5), (0, 1, 2)]) + self.__bias.value
        out = (255 * out).astype('uint8')
        y = Tensor(out)
        y.append_bp_cache(self.backward, {'x': x, 'w': self.__filter, 'b': self.__bias})
        return y

    @staticmethod
    def backward(**kwargs: Tensor) -> None:
        y = kwargs['y']
        x = kwargs['x']
        w = kwargs['w']
        b = kwargs['b']
        out_height, out_width = y.grad.shape[1:-1]
        kernel_height, kernel_width = w.value.shape[1:-1]
        b.grad += y.grad.sum(axis=(0, 1, 2))
        w.grad = np.zeros(w.value.shape)
        for height in range(kernel_height):
            for width in range(kernel_width):
                w.grad[:, height, width, :] = np.tensordot(
                    y.value, x.value[:, height:height + out_height, width:width + out_width, :],
                    axes=([0, 1, 2], [0, 1, 2]))

    @staticmethod
    def __split_by_strides(x: np.ndarray, kernel_height: int, kernel_width: int, strides: int) -> np.ndarray:
        num, height, width, channel = x.shape
        out_height = (height - kernel_height) // strides + 1
        out_width = (width - kernel_width) // strides + 1
        shape = num, out_height, out_width, kernel_height, kernel_width, channel
        strides = (x.strides[0], x.strides[1] * strides, x.strides[2] * strides, *x.strides[1:])
        out = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return out
