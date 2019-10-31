from tensorflow import keras
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]

model = keras.models.Sequential([
    keras.layers.Dense(10, activation=keras.activations.relu),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)

model.fit(train_x, train_y, batch_size=32, epochs=3, validation_split=0.2, shuffle=True)
model.evaluate(test_x, test_y, verbose=1)