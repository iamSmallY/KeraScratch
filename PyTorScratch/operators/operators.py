from abc import ABCMeta, abstractmethod

from PyTorScratch.tensor import Tensor


class Operator(metaclass=ABCMeta):
    """神经网络算子抽象类。

    对于一个神经网络中的某一算子而言，仅需要一个前向传播，以及一个反向传播方法。
    """

    def args(self):
        """获取算子全部参数方法。

        获取当前算子所含的全部张量即可。
        """
        res = []
        for name in vars(self).keys():
            if isinstance(getattr(self, name), Tensor):
                res.append(getattr(self, name))
        return res

    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """计算前向传播方法。

        将张量 x 通过该算子进行前向传播，计算其结果，并返回结果张量。

        Args:
            x: 用于计算前向传播的张量

        Returns:
            计算得到的结果张量。
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        """计算反向传播方法。

        根据算子计算得到的张量 y 的值，更新张量 x 的梯度值。\n
        该函数应用于在前向传播时被放入张量的 bp_cache 中。

        Args:
            x: 反向传播时输出的张量
            y: 反向传播时输入的张量
        """
        raise NotImplementedError()
