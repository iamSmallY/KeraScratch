"""张量类，基于 numpy 实现。"""
from __future__ import annotations

import numpy as np
from typing import Dict, Callable, List, NoReturn, Optional, Tuple


class Tensor(object):
    """Tensor 张量类

    用于模型的底层计算，基于 numpy 实现。用于存储原始值、梯度，和计算、缓存反向传播时的梯度。

    Attributes:
        __value: 该张量的值 (getter)
        __grad: 该张量对应的梯度 (getter, setter)
    """

    def __init__(self, value: np.ndarray) -> None:
        """初始化 Tensor 对象

        为该张量进行初始化，用传入的 value 初始化 __value，
        并将梯度都初始化为与 value 形状相同的 0 向量，将反向传播结果设置为空列表。

        Args:
            value: 用于初始化该张量的值
        """
        self.__value: np.ndarray = np.copy(value)
        self.__grad: np.ndarray = np.zeros(value.shape)

    @property
    def value(self) -> np.ndarray:
        """获取 value 值方法。"""
        return self.__value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self.__value = np.copy(value)

    @property
    def grad(self) -> np.ndarray:
        """获取 grad 值方法。"""
        return self.__grad

    @grad.setter
    def grad(self, value: np.ndarray) -> None:
        """设置 grad 值方法"""
        self.__grad = np.copy(value)

    def zero_grad(self) -> None:
        """递归地清空已有梯度。"""
        self.__grad = np.zeros(self.__grad.shape)

    def __str__(self) -> str:
        return f'Value:{str(self.__value)}\nGradient:{str(self.__grad)}'

