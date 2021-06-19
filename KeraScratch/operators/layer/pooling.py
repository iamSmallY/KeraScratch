from .layer import Layer
from KeraScratch.tensor import Tensor


class MaxPoolingOperator(Layer):
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


class MeanPoolingOperator(Layer):
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
