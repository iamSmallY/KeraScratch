import numpy as np

from .loss import Loss
from KeraScratch.tensor import Tensor


class CrossEntropyLoss(Loss):
    """交叉熵损失函数类。

    内部整合了 softmax 函数，便于计算，在使用模型时最后一层无需 softmax 函数。
    """

    @staticmethod
    def softmax(x: Tensor) -> np.ndarray:
        """计算 softmax 函数方法。

        Args:
            x: 输入张量

        Returns:
            softmax(x)
        """
        e = np.exp(x.value)
        return e / np.sum(e)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        target: Tensor = kwargs['target']
        s = self.softmax(x)
        loss = -np.log(s[0][target.value[0]])
        y = Tensor(np.array([loss]))
        return y

    def backward(self, x: Tensor, y: Tensor, **kwargs) -> None:
        target: Tensor = kwargs['target']
        x.grad = self.softmax(x)
        x.grad[0][target.value[0]] -= 1
