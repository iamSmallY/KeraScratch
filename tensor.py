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
        __bp_cache: 用于计算、缓存反向传播的结果，存储反向传播时所用的计算函数与该函数的参数
    """

    def __init__(self, value: np.ndarray) -> None:
        """初始化 Tensor 对象

        为该张量进行初始化，用传入的 value 初始化 __value，
        并将梯度都初始化为与 value 形状相同的 0 向量，将反向传播结果设置为空列表。

        Args:
            value: 用于初始化该张量的值
        """
        self.__value: np.ndarray = value
        self.__grad: np.ndarray = np.zeros(value.shape)
        self.__bp_cache: List[Tuple[Callable[[Dict[str, Tensor]], NoReturn],
                                    Dict[str, Tensor]]] = []

    @property
    def value(self) -> np.ndarray:
        """获取 value 值方法。"""
        return self.__value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self.__value = value

    @property
    def grad(self) -> np.ndarray:
        """获取 grad 值方法。"""
        return self.__grad

    @grad.setter
    def grad(self, value: np.ndarray) -> None:
        """设置 grad 值方法"""
        self.__grad = value

    def append_bp_cache(self, back_func: Callable[[Dict[str, Tensor], Optional[str]], NoReturn],
                        kwargs: Dict[str, Tensor]) -> None:
        """添加 bp_cache 方法。

        Args:
            back_func: 反向传播所用函数，
            kwargs: bp_func 的参数
        """
        self.__bp_cache.append((back_func, kwargs))

    def backward(self) -> None:
        """递归计算反向传播结果。"""
        for back_func, kwargs in self.__bp_cache:
            back_func(y=self, **kwargs)
            for name, t in kwargs.items():
                t.backward()

    def zero_grad(self) -> None:
        """递归地清空已有梯度。"""
        self.__grad = np.zeros(self.__grad.shape)
        for back_func, kwargs in self.__bp_cache:
            for name, t in kwargs.items():
                t.zero_grad()

    def __str__(self) -> str:
        return f'Value:{str(self.__value)}\nGradient:{str(self.__grad)}'

