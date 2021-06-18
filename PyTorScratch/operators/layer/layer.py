from abc import abstractmethod

from PyTorScratch.operators import Operator


class Layer(Operator):
    """层级算子抽象类。"""
    @abstractmethod
    def zero_grad(self) -> None:
        """清空自身参数的缓存梯度方法。"""
        raise NotImplemented()
