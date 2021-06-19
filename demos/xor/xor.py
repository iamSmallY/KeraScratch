from matplotlib import pyplot as plt
import numpy as np

from KeraScratch import model, optimizers
from KeraScratch.operators import activation, layer, loss

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

epoch = 3000
history = md.fit(train_data, epoch=epoch)
print(history['train'])

for data in train_data:
    x, _ = data
    print(md.predict(x))

x_range = range(epoch)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_range, history['train']['accuracy'], label='accuracy')
ax.set_ylabel('accuracy')
ax.legend()

ax2 = ax.twinx()
ax2.plot(x_range, history['train']['loss'], 'r', label='loss')
ax2.set_ylabel('loss')
ax2.legend()

plt.show()
