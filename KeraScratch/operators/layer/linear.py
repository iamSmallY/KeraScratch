import numpy as np
from typing import Optional

from .layer import Layer
from KeraScratch.tensor import Tensor


class LinearOperator(Layer):
    """全连接算子。"""

    def __init__(self, out_dim: int) -> None:
        """初始化全连接层方法。

        全连接算子计算方式为 y = wx + b.\n
        对于 in_dim，我们在第一次前向传播时根据 x 的形状来自动生成，所以参数仅需 out_dim.

        Args:
            out_dim: 需要张量的维度
        """
        self.__out_dim = out_dim
        self.__w: Optional[Tensor] = None
        self.__b: Optional[Tensor] = None

    def zero_grad(self) -> None:
        if self.__w is None:
            return
        self.__w.grad = np.zeros(self.__w.grad.shape)
        self.__b.grad = np.zeros(self.__b.grad.shape)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.__w is None:
            self.__w = Tensor(np.random.randn(x.value.shape[-1], self.__out_dim) * 0.1)
            self.__b = Tensor(np.random.randn(x.value.shape[0], self.__out_dim) * 0.1)

        y = Tensor(np.dot(x.value, self.__w.value) + self.__b.value)
        return y

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        x.grad += np.dot(y.grad, self.__w.value.T)
        self.__w.grad += np.dot(x.value.T, y.grad)
        self.__b.grad += y.grad
