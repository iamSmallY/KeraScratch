from abc import ABCMeta

from PyTorScratch.operators import Operator


class Loss(Operator, metaclass=ABCMeta):
    """损失算子抽象类。"""
    pass
