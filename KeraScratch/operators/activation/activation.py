from abc import abstractmethod

from KeraScratch.operators.operators import Operator
from KeraScratch.tensor import Tensor


class ActivationOperator(Operator):
    """激活函数算子抽象类

    扩展了 Operator 类，增加了激活函数算子所需要用到的计算激活函数的方法。
    """

    @staticmethod
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """计算张量经过激活函数后的值"""
        raise NotImplementedError()

