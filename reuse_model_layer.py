from tensorflow import keras
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]


# define your reusable layers in here
l1 = keras.layers.Dense(10, activation=keras.activations.relu)


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = l1  # this is a reusable layer
        self.l2 = keras.layers.Dense(1)   # this is NOT a reusable layer

    def call(self, x, training=None, mask=None):
        x = self.l1(x)
        x = self.l2(x)
        return x


model1 = Model()
model2 = Model()

model1.build((None, 1))
model2.build((None, 1))

model1.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)

# train model1 for a while
model1.fit(train_x, train_y, batch_size=32, epochs=3, validation_split=0.2, shuffle=True)
print("l1 is reused: ", np.all(model1.l1.get_weights()[0] == model2.l1.get_weights()[0]))
print("l2 is reused: ", np.all(model1.l2.get_weights()[0] == model2.l2.get_weights()[0]))