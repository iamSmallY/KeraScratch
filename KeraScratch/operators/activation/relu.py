import numpy as np

from .activation import ActivationOperator
from KeraScratch.tensor import Tensor


class ReLUOperator(ActivationOperator):
    """ReLU 算子。"""

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor((x.value > 0).astype(np.float32) * x.value)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        y = self(x)
        return y

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        x.grad += y.grad * (y.value > 0).astype(np.float32)
