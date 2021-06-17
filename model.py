"""模型模块"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from loss import Loss
from operators import Operator
from optimizers import Optimizer
from tensor import Tensor


class Model(metaclass=ABCMeta):
    """模型抽象类。"""

    def args(self):
        """获取模型全部参数方法。

        参数即为张量的梯度，于是递归获取每个算子对应的梯度即可。
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
        pass

    @abstractmethod
    def predict(self, data: Tensor) -> Tensor:
        """利用模型进行预测。

        Args:
            data: 待预测的张量

        Returns:
            预测的结果张量
        """
        pass

    @abstractmethod
    def evaluate(self, data: List[List[Tensor]]) -> Dict[str, float]:
        """评估模型方法。

        传入一组数据，返回该组数据的正确率与损失值，用来评估模型效果。

        Args:
            data: 传入的数据集

        Returns:
            该组数据预测结果的正确率与损失值，
        """
        pass

    @abstractmethod
    def add(self, operator: Operator) -> None:
        """为神经网络添加一个算子。

        Args:
            operator: 新算子
        """
        pass

    @abstractmethod
    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        """编译模型方法。

        为模型添加损失函数与优化器，使得模型可以开始训练。

        Args:
            loss: 损失函数计算对象
            optimizer: 优化器对象
        """
        pass


class Sequential(Model):
    """线性组合模型类。

    该模型将算子通过线性组合简单的连接在一起。
    """

    def __init__(self, operators: List[Operator]) -> None:
        """初始化线性组合模型方法。

        Args:
            operators: 算子列表，按从前往后的运算顺序存储
        """
        self.__operators = operators
        self.__loss: Optional[Loss] = None
        self.__optimizer: Optional[Optimizer] = None

    def fit(self, train_data: List[List[np.ndarray]], val_data: List[List[np.ndarray]] = None,
            epoch: int = 5) -> Dict[str, Dict[str, List[float]]]:
        res = {
            'train': {
                'accuracy': [],
                'loss': []
            }
        }
        if val_data is not None:
            res['val'] = {
                'accuracy': [],
                'loss': []
            }
        for _ in range(epoch):
            temp_train_loss = []
            temp_val_loss = []
            for data in train_data:
                x, y = data
                x, y = Tensor(x), Tensor(y)
                train_loss = self.__calculate(x, y)
                temp_train_loss.append(train_loss.value[0])
                train_loss.backward()
                self.__optimizer.step()
            if val_data is not None:
                for data in val_data:
                    x, y = data
                    x, y = Tensor(x), Tensor(y)
                    temp_val_loss.append(self.__calculate(x, y).value[0])
            res['train']['loss'].append(np.mean(np.array(temp_train_loss)))
            if val_data is not None:
                res['val']['loss'].append(np.mean(np.array(temp_val_loss)))
        return res

    def predict(self, data: Tensor) -> Tensor:
        pass

    def evaluate(self, data: List[List[Tensor]]) -> Dict[str, float]:
        pass

    def add(self, operator: Operator) -> None:
        """添加算子方法。

        在模型的算子列表最后添加新传入的算子。

        Args:
            operator: 待添加的算子
        """
        self.__operators.append(operator)

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        self.__loss = loss
        self.__optimizer = optimizer

    def __forward(self, x: Tensor) -> Tensor:
        for operator in self.__operators:
            x = operator.forward(x)
        return x

    @staticmethod
    def __backward(y: Tensor) -> None:
        y.backward()

    def __calculate(self, x: Tensor, y: Tensor) -> Tensor:
        """计算损失值方法。"""
        pred = self.__forward(x)
        loss = self.__loss.calculate(pred, y)
        pred.zero_grad()
        return loss
