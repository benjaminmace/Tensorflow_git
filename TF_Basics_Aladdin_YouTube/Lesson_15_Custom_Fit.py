import os
import matplotlib.pyplot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, 3, padding='same'),
        layers.ReLU(),
        layers.Conv2D(128, 3, padding='same'),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(10)
    ],
    name='model'
)

class CustomFit(keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model

    def comile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss


    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)

        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))
        acc_metrics.update_state(y, y_pred)

        return {'loss': loss, 'accuracy':acc_metrics.result()}

    def test_step(self, data):
        x, y = data

        y_pred = self.model(x, training=False)
        loss = self.loss(y, y_pred)
        acc_metrics.update_state(y, y_pred)

        return {'loss': loss, 'accuracy': acc_metrics.result()}



acc_metrics = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

training = CustomFit(model)

training.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)


training.fit(x_train, y_train, batch_size=32, epochs=2)
training.evaluate(x_test, y_test, batch_size=32)