import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]


def create_model():
    return keras.models.Sequential([
        keras.layers.Dense(10, activation=keras.activations.relu, input_shape=(1, )),
        keras.layers.Dense(1),
    ])


model = create_model()
model.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
)

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
ckpt_dir = os.path.dirname(checkpoint_path)
os.makedirs(ckpt_dir, exist_ok=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    period=5
)

# save ckpt
model.save_weights(checkpoint_path.format(epoch=0))


history = model.fit(
    train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y), shuffle=True,
    callbacks=[cp_callback, ]  # save when callback
)

# restore ckpt
latest_model = tf.train.latest_checkpoint(ckpt_dir)
model2 = create_model()
model2.load_weights(latest_model)
model2.compile(
    optimizer=keras.optimizers.SGD(0.01),
    loss=keras.losses.MeanSquaredError(),
)
loss = model2.evaluate(test_x, test_y, verbose=2)
print("Restored ckpt model, loss: {:.2f}".format(loss))


# save pb
final_path = "training/final_model"
model.save(final_path)

# restore
model3 = keras.models.load_model(final_path)
loss = model3.evaluate(test_x, test_y, verbose=2)
print("Restored pb model, loss: {:.2f}".format(loss))