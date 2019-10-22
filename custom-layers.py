import tensorflow as tf
from tensorflow import keras
import numpy as np


data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]


class MyLayer(keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=[int(input_shape[-1]), self.num_outputs],
            dtype=tf.float32,
            initializer=keras.initializers.RandomNormal(),
        )
        self.b = self.add_weight(
            name="b",
            shape=[1, self.num_outputs],
            dtype=tf.float32,
            initializer=keras.initializers.Constant(0.1),
        )

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


model = keras.models.Sequential([
    MyLayer(10),
    keras.layers.Dense(1),
])

model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)

model.fit(train_x, train_y, batch_size=32, epochs=3, validation_split=0.2, shuffle=True)
model.evaluate(test_x, test_y, verbose=1)