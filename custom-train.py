import tensorflow as tf
from tensorflow import keras
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = keras.layers.Dense(10, activation=keras.activations.relu, input_shape=(1, ))
        self.l2 = keras.layers.Dense(1)

    def call(self, x, training=None, mask=None):
        x = self.l1(x)
        x = self.l2(x)
        return x


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_ = model(x)
        loss = loss_func(y, y_)
    grad = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grad, model.trainable_variables))
    return loss


model = Model()
opt = tf.optimizers.SGD(0.1)
loss_func = keras.losses.MeanSquaredError()


for t in range(100):
    loss = train_step(data_x, data_y)
    if t % 10 == 0:
        print("loss={:.2f}".format(loss.numpy()))