from typing import Dict, List, Optional

import numpy as np

from .model import Model
from KeraScratch.operators import layer, loss, Operator
from KeraScratch.optimizers import Optimizer
from KeraScratch.tensor import Tensor


class SequentialModel(Model):
    """线性组合模型类。

    该模型将算子通过线性组合简单的连接在一起。
    """

    def __init__(self, operators: List[Operator]) -> None:
        """初始化线性组合模型方法。

        Args:
            operators: 算子列表，按从前往后的运算顺序存储
        """
        self.__operators = operators
        self.__optimizer: Optional[Optimizer] = None

    def fit(self, train_data: List[List[np.ndarray]], val_data: List[List[np.ndarray]] = None,
            epoch: int = 5) -> Dict[str, Dict[str, List[float]]]:
        assert isinstance(self.__operators[-1], loss.Loss)
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
        for i in range(epoch):
            temp_train_acc, temp_train_loss = [], []
            temp_val_acc, temp_val_loss = [], []
            for data in train_data:
                pred_list = self.__epoch_forward(data, temp_train_acc, temp_train_loss)

                _, y = data
                self.__backward(pred_list, Tensor(y))
                self.__optimizer.step(self.args())

            if val_data is not None:
                for data in val_data:
                    self.__epoch_forward(data, temp_val_acc, temp_val_loss)

            res['train']['accuracy'].append(np.mean(np.array(temp_train_acc)))
            res['train']['loss'].append(np.mean(np.array(temp_train_loss)))
            string = f'epoch: {i + 1}/{epoch}, train_acc: {np.mean(np.array(temp_train_acc))}  ' \
                     f'train_loss: {np.mean(np.array(temp_train_loss))}'
            if val_data is not None:
                res['val']['accuracy'].append(np.mean(np.array(temp_val_acc)))
                res['val']['loss'].append(np.mean(np.array(temp_val_loss)))
                string += f', test_acc: {np.mean(np.array(temp_val_acc))}, ' \
                          f'test_loss: {np.mean(np.array(temp_val_loss))}'
            print(string)
        return res

    def predict(self, data: np.ndarray) -> np.ndarray:
        assert isinstance(self.__operators[-1], loss.Loss)
        assert self.__optimizer is not None
        pred = self.__forward(Tensor(data), predict=True)[-1]
        if isinstance(self.__operators[-1], loss.CrossEntropyLoss):
            # 对于交叉熵损失函数，预测结果需要添加一个 softmax 运算
            pred = loss.CrossEntropyLoss.softmax(pred)
            return np.argmax(pred)
        return pred.value

    def evaluate(self, data: List[List[np.ndarray]]) -> Dict[str, np.ndarray]:
        assert isinstance(self.__operators[-1], loss.Loss)
        assert self.__optimizer is not None
        acc_list, loss_list = [], []
        for d in data:
            self.__epoch_forward(d, acc_list, loss_list)
        return {
            'accuracy': np.mean(np.array(acc_list)),
            'loss': np.mean(np.array(loss_list))
        }

    def compile(self, optimizer: Optimizer) -> None:
        self.__optimizer = optimizer

    def add(self, operator: Operator) -> None:
        """添加算子方法。

        在模型的算子列表最后添加新传入的算子。

        Args:
            operator: 待添加的算子
        """
        self.__operators.append(operator)

    def __forward(self, x: Tensor, predict=False, expect_out: Tensor = None) -> List[Tensor]:
        res = [x]
        length = len(self.__operators)
        if predict:
            length -= 1
        for i in range(length):
            if isinstance(self.__operators[i], loss.CrossEntropyLoss):
                res.append(self.__operators[i].forward(res[-1], target=expect_out))
                continue
            res.append(self.__operators[i].forward(res[-1]))
        return res

    def __backward(self, pred_list: List[Tensor], y: Tensor) -> None:
        assert len(pred_list) == len(self.__operators) + 1
        for i in range(len(self.__operators) - 1, -1, -1):
            if isinstance(self.__operators[i], layer.Layer):
                self.__operators[i].zero_grad()
            if isinstance(self.__operators[i], loss.CrossEntropyLoss):
                self.__operators[i].backward(pred_list[i], pred_list[i + 1], target=y)
            else:
                self.__operators[i].backward(pred_list[i], pred_list[i + 1])

    def __epoch_forward(self, data: List[np.ndarray], acc_list: List[int], loss_list: List[Tensor]) -> List[Tensor]:
        """计算一次迭代中的前向传播的方法。"""
        x, y = data
        x, y = Tensor(x), Tensor(y)
        pred_list = self.__forward(x, expect_out=y)
        if isinstance(self.__operators[-1], loss.CrossEntropyLoss):
            predict = np.argmax(loss.CrossEntropyLoss.softmax(pred_list[-2]))
        else:
            predict = pred_list[-2].value[0]

        acc_list.append(1 if y.value[0] == predict else 0)
        loss_list.append(pred_list[-1].value[0])

        return pred_list
