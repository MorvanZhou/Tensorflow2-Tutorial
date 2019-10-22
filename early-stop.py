from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]

model = keras.models.Sequential([
    keras.layers.Dense(10, activation=keras.activations.relu, input_shape=(1, )),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)
history = model.fit(
    train_x, train_y, batch_size=32, epochs=100, validation_split=0.2, shuffle=True,
    callbacks=[early_stop, ]
)

plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("val_loss")
plt.show()