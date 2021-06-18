import numpy as np

from .layer import Layer
from PyTorScratch.tensor import Tensor


class FlattenOperator(Layer):
    """拉平算子。"""
    def zero_grad(self) -> None:
        pass

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out_dim = 1
        for i in range(1, len(x.value.shape)):
            out_dim *= x.value.shape[i]
        return Tensor(x.value.reshape((x.value.shape[0], out_dim)))

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        x.grad = y.grad.reshape(x.grad.shape)
