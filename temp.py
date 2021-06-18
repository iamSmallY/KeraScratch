import cv2
import numpy as np

from PyTorScratch.operators.operators import Conv2DOperator
from PyTorScratch.tensor import Tensor

img = cv2.imread('035.jpeg')
cv2.imshow('1', img)
conv2D = Conv2DOperator(3, 3)

a = Tensor(np.array([img]))
out = conv2D.forward(a)
print(out.value[0, :, :, :].shape)
cv2.imshow('2', out.value[0, :, :, :])
out.backward()
cv2.imshow('3', a.value[0, :, :, :])
print(a.grad)
cv2.waitKey(0)
