from abc import ABCMeta

from KeraScratch.operators import Operator
from KeraScratch.tensor import Tensor


class Layer(Operator, metaclass=ABCMeta):
    """层级算子抽象类。"""
    def zero_grad(self) -> None:
        """清空自身参数的缓存梯度方法。"""
        for name in vars(self).keys():
            if isinstance(getattr(self, name), Tensor):
                getattr(self, name).zero_grad()
