import numpy as np
from tensor import Tensor
import unittest


class MyTestCase(unittest.TestCase):
    def test_tensor(self):
        # 测试初始化
        ts = Tensor(np.array([1, 2, 3]))
        self.assertTrue(np.array_equal(ts._Tensor__value, np.array([1, 2, 3])))


if __name__ == '__main__':
    unittest.main()
