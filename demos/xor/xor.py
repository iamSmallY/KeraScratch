import numpy as np

from PyTorScratch import model, optimizers
from PyTorScratch.operators import activation, layer, loss

train_data = [
    [np.array([[0, 0]]), np.array([0])],
    [np.array([[0, 1]]), np.array([1])],
    [np.array([[1, 0]]), np.array([1])],
    [np.array([[1, 1]]), np.array([0])]
]

md = model.SequentialModel([
    layer.LinearOperator(2),
    activation.SigmoidOperator(),
    layer.LinearOperator(2),
    loss.CrossEntropyLoss()
])
md.compile(optimizers.SGD(learning_rate=1e-1, momentum=0))

history = md.fit(train_data, epoch=5000)
print(history['train'])

for data in train_data:
    x, _ = data
    print(md.predict(x))
