from abc import ABCMeta, abstractmethod
from typing import Dict, List

import numpy as np

from PyTorScratch.operators import Operator
from PyTorScratch.optimizers import Optimizer
from PyTorScratch.tensor import Tensor


class Model(metaclass=ABCMeta):
    """模型抽象类。"""

    def args(self):
        """获取模型全部参数方法。

        参数即为张量的梯度，于是递归获取每个算子对应的张量即可。
        """
        res = []
        for name, value in vars(self).items():
            if isinstance(value, list):
                for operator in value:
                    res += operator.args()
            if isinstance(value, Operator):
                res += value.args()
        return res

    @abstractmethod
    def fit(self, train_data: List[List[np.ndarray]], val_data: List[List[np.ndarray]] = None,
            epoch: int = 5) -> Dict[str, Dict[str, List[float]]]:
        """模型拟合方法。

        利用输入的 x 张量去拟合 y 张量，并输出准确率、损失值。

        Args:
            train_data: 训练集数据
            val_data: 验证集数据
            epoch: 迭代次数

        Returns:
            由训练集以及验证集的每轮准确率 accuracy 对应的元组及每轮损失值 loss 对应的元组所组成的字典
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data: Tensor) -> np.ndarray:
        """利用模型进行预测。

        Args:
            data: 待预测的张量

        Returns:
            预测的结果
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, data: List[List[Tensor]]) -> Dict[str, float]:
        """评估模型方法。

        传入一组数据，返回该组数据的正确率与损失值，用来评估模型效果。

        Args:
            data: 传入的数据集

        Returns:
            该组数据预测结果的正确率与损失值，
        """
        raise NotImplementedError()

    @abstractmethod
    def compile(self, optimizer: Optimizer) -> None:
        """编译模型方法。

        为模型添加优化器，使得模型可以开始训练。

        Args:
            optimizer: 优化器对象
        """
        raise NotImplementedError()
