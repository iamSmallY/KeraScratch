from functools import reduce
import numpy as np
from typing import Optional

from .layer import Layer
from KeraScratch.tensor import Tensor


class Conv2DOperator(Layer):
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

    def zero_grad(self) -> None:
        self.__filter.grad = np.zeros(self.__filter.grad.shape)
        self.__bias.grad = np.zeros(self.__bias.grad.shape)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.__filter is None:
            kernel_shape = self.__out_channel, self.__kernel_size, self.__kernel_size, x.value.shape[-1]
            self.__filter = Tensor(np.random.randn(*kernel_shape) *
                                   (2 / reduce(lambda _x, _y: _x * _y, kernel_shape[:-1]) ** 0.5))
            self.__bias = Tensor(np.zeros(self.__out_channel))

        x.value = self.padding(x.value)
        x_split = self.__split_by_strides(x.value)
        a = np.tensordot(x_split, self.__filter.value, axes=[(3, 4, 5), (1, 2, 3)]) + self.__bias.value
        return Tensor(a)

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        batch_size = y.grad.shape[0]
        x_split = self.__split_by_strides(x.value)
        self.__filter.grad = np.tensordot(y.grad, x_split, [(0, 1, 2), (0, 1, 2)]) / batch_size
        self.__bias.grad = np.reshape(y.grad, [batch_size, -1, self.__out_channel]).sum(axis=(0, 1)) / batch_size

        if self.__strides > 1:
            temp = np.zeros(y.grad.shape)
            temp[:, ::self.__strides, ::self.__strides, :] = y.grad
            y.grad = temp
        y_pad = self.padding(y.grad, False)
        filter_rot = self.__filter.value[:, ::-1, ::-1, :].swapaxes(0, 3)
        y_split = self.__split_by_strides(y_pad)
        x.grad += np.tensordot(y_split, filter_rot, axes=[(3, 4, 5), (1, 2, 3)])

    def padding(self, x: np.ndarray, forward=True) -> np.ndarray:
        """对输入矩阵进行填充方法。"""
        p = self.__kernel_size // 2 if self.__padding == 'same' else self.__kernel_size - 1
        if forward:
            return x if self.__padding == 'valid' else np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        else:
            return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')

    def __split_by_strides(self, x: np.ndarray) -> np.ndarray:
        num, height, width, channel = x.shape
        out_height = (height - self.__kernel_size) // self.__strides + 1
        out_width = (width - self.__kernel_size) // self.__strides + 1
        shape = num, out_height, out_width, self.__kernel_size, self.__kernel_size, channel
        strides = (x.strides[0], x.strides[1] * self.__strides, x.strides[2] * self.__strides, *x.strides[1:])
        out = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return out
