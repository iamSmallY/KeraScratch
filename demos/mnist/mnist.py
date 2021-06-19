from matplotlib import pyplot as plt
import numpy as np
from random import shuffle

from KeraScratch import model, optimizers
from KeraScratch.operators import activation, layer, loss


def get_images(path, split=0):
    ds = np.load(path)
    x = ds['X']
    y = ds['Y']
    data = []
    for i in range(x.shape[0]):
        data.append([np.array([x[i, :].astype(np.float64) / 255]), np.array([np.argmax(y[i, :])])])
    if split == 0:
        return data
    shuffle(data)
    return data[:int(split*len(data))], data[int(split*len(data)):]


train_data, val_data = get_images('image/trainset.npz', 0.9)
test_data = get_images('image/testset.npz')


cnn = model.SequentialModel([
    layer.Conv2DOperator(8, 5, padding='same'),
    activation.ReLUOperator(),
    layer.MaxPoolingOperator(2),
    layer.Conv2DOperator(16, 5, padding='same'),
    activation.ReLUOperator(),
    layer.MeanPoolingOperator(2),
    layer.FlattenOperator(),
    layer.LinearOperator(64),
    activation.ReLUOperator(),
    layer.LinearOperator(10),
    loss.CrossEntropyLoss()
])
cnn.compile(optimizers.SGD(learning_rate=1e-3, momentum=0))

epoch = 20
history = cnn.fit(train_data, val_data, epoch=epoch)

x_range = range(epoch)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_range, history['train']['accuracy'], label='train acc')
ax.plot(x_range, history['val']['accuracy'], label='test acc')
ax.set_ylabel('accuracy')
ax.legend(loc='upper left')

ax2 = ax.twinx()
ax2.plot(x_range, history['train']['loss'], label='train loss')
ax2.plot(x_range, history['val']['loss'], label='test loss')
ax2.set_ylabel('loss')
ax2.legend(loc='upper right')

plt.show()

test_his = cnn.evaluate(test_data)
print(test_his)
