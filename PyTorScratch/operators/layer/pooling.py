from abc import ABCMeta

from .layer import Layer
from PyTorScratch.tensor import Tensor


class PoolingOperator(Layer, metaclass=ABCMeta):
    """池化层抽象类，所有池化层步长（strides）默认为池化大小。"""
    def zero_grad(self) -> None:
        pass


class MaxPoolingOperator(PoolingOperator):
    """最大池化算子。"""
    def __init__(self, size: int):
        self.__size = size
        self.__index = None

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out = x.value.reshape((x.value.shape[0], x.value.shape[1] // self.__size, self.__size,
                              x.value.shape[2] // self.__size, self.__size, x.value.shape[3]))
        out = out.max(axis=(2, 4))
        self.__index = out.repeat(self.__size, axis=1).repeat(self.__size, axis=2) == x
        return Tensor(out)

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        x.grad += y.grad.repeat(self.__size, axis=1).repeat(self.__size, axis=2) * self.__index


class MeanPoolingOperator(PoolingOperator):
    """均值池化算子。"""
    def __init__(self, size):
        self.__size = size

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out = x.value.reshape((x.value.shape[0], x.value.shape[1] // self.__size, self.__size,
                              x.value.shape[2] // self.__size, self.__size, x.value.shape[3]))
        return Tensor(out.mean(axis=(2, 4)))

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        temp = y.grad / self.__size ** 2
        x.grad += temp.repeat(self.__size, axis=1).repeat(self.__size, axis=2)
