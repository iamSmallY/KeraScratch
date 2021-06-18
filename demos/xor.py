import numpy as np

from PyTorScratch import model, loss, optimizers
from PyTorScratch.operators import operators, activation

train_data = [
    [np.array([[0, 0]]), np.array([0])],
    [np.array([[0, 1]]), np.array([1])],
    [np.array([[1, 0]]), np.array([1])],
    [np.array([[0, 0]]), np.array([0])]
]

md = model.Sequential([
    operators.LinearOperator(2, 2),
    activation.SigmoidOperator(),
    operators.LinearOperator(2, 2)
])
md.compile(loss.CrossEntropyLoss(), optimizers.SGD(md.args(), learning_rate=1e-1, momentum=0))

history = md.fit(train_data, epoch=1000)
print(history['train'])
