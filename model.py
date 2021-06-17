"""模型模块"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from loss import Loss, CrossEntropyLoss
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
        assert self.__loss is not None
        assert self.__optimizer is not None
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
            temp_train_acc, temp_train_loss = [], []
            temp_val_acc, temp_val_loss = [], []
            for data in train_data:
                x, y = data
                x, y = Tensor(x), Tensor(y)
                acc, loss = self.__calculate(x, y)
                temp_train_acc.append(acc)
                temp_train_loss.append(loss.value[0])
                loss.backward()
                self.__optimizer.step()
            if val_data is not None:
                for data in val_data:
                    x, y = data
                    x, y = Tensor(x), Tensor(y)
                    acc, loss = self.__calculate(x, y)
                    temp_val_acc.append(acc)
                    temp_val_loss.append(loss.value[0])
            res['train']['accuracy'].append(sum(temp_train_acc) / len(temp_train_acc))
            res['train']['loss'].append(np.mean(np.array(temp_train_loss)))
            if val_data is not None:
                res['val']['accuracy'].append(sum(temp_val_acc) / len(temp_val_acc))
                res['val']['loss'].append(np.mean(np.array(temp_val_loss)))
        return res

    def predict(self, data: Tensor) -> Tensor:
        assert self.__loss is not None
        assert self.__optimizer is not None
        pred = self.__forward(data)
        if isinstance(self.__optimizer, CrossEntropyLoss):
            # 对于交叉熵损失函数，预测结果需要添加一个 softmax 运算
            pred = CrossEntropyLoss.softmax(pred)
            return Tensor(np.argmax(pred))
        return pred

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

    def __calculate(self, x: Tensor, y: Tensor) -> Tuple[int, Tensor]:
        """计算损失值方法。"""
        pred = self.__forward(x)
        if isinstance(self.__loss, CrossEntropyLoss):
            predict = np.argmax(CrossEntropyLoss.softmax(pred))
        else:
            predict = pred.value[0]
        acc = 1 if y.value[0] == predict else 0
        loss = self.__loss.calculate(pred, y)
        pred.zero_grad()
        return acc, loss
