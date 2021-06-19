import numpy as np

from .activation import ActivationOperator
from KeraScratch.tensor import Tensor


class SigmoidOperator(ActivationOperator):
    """Sigmoid 算子。"""

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x.value)))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        y = self(x)
        return y

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        x.grad += y.grad * self(x).value * (1 - self(x).value)
