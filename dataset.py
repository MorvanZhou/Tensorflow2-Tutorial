import tensorflow as tf
from tensorflow import keras
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise


train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(1000).batch(32)

model = keras.models.Sequential([
    keras.layers.Dense(10, activation=keras.activations.relu, input_shape=(1, )),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)

for epoch in range(3):
    train_losses = []
    for bx, by in train_ds:
        out = model.train_on_batch(bx, by)
        # out = [loss, metrics]
        train_losses.append(out[0])

    test_losses = []
    for bx, by in test_ds:
        loss = model.evaluate(bx, by, verbose=0)
        test_losses.append(loss)
    print("train loss={:.2f} | test loss={:.2f}".format(np.mean(train_losses), np.mean(test_losses)))
