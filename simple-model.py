import tensorflow as tf
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise


class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.w = tf.Variable(0.1, dtype=tf.float32)
        self.b = tf.Variable(0.1, dtype=tf.float32)

    def call(self, x):
        return self.w * x + self.b


model = Model()
var_list = [model.w, model.b]
opt = tf.optimizers.SGD(0.1)

for t in range(100):
    with tf.GradientTape() as tape:
        y_ = model.call(data_x)
        loss = tf.reduce_mean(tf.square(data_y - y_))

    grad = tape.gradient(loss, var_list)
    opt.apply_gradients(zip(grad, var_list))
    if t % 10 == 0:
        print("loss={:.2f} | w={:.2f} | b={:.2f}".format(
            loss, model.w.numpy(), model.b.numpy())
        )
