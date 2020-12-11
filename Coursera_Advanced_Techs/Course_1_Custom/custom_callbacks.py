import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import datetime
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import imageio

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

plt.rc('font', size=20)
plt.rc('figure', figsize=(15, 3))


def display_digits(inputs, outputs, ground_truth, epoch, n=10):
    plt.clf()

    plt.yticks([])
    plt.grid(None)
    inputs = np.reshape(inputs, [n, 28, 28])
    inputs = np.swapaxes(inputs, 0, 1)
    inputs = np.reshape(inputs, [28, 28 * n])
    plt.imshow(inputs)
    plt.xticks([28 * x + 14 for x in range(n)], outputs)
    for i, t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if outputs[i] == ground_truth[i]:
            t.set_color('green')
        else:
            t.set_color('red')
    plt.grid(None)

GIF_PATH = './animation.gif'

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print('Training: batch {} at {}'.format(
            batch, datetime.datetime.now().time()
        ))

    def on_batch_end(self, batch, logs=None):
        print('Training: batch {} ends at {}'.format(
            batch, datetime.datetime.now().time()
        ))

class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs['val_loss'] / logs['loss']
        print('Epoch: {}, Val/Train loss ration: {:.2f}'.format(epoch, ratio))

        if ratio > self.threshold:
            print('Stopping training')
            self.model.stop_training = True

class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, display_freq = 10, n_samples = 10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples

    def on_epoch_end(self, epoch, logs=None):
        # Random sample data
        indexes = np.random.choice(len(self.inputs), size=self.n_samples)
        x_test, y_test = self.inputs[indexes], self.ground_truth[indexes]
        predictions = np.argmax(self.model.predict(x_test), axis=1)

        display_digits(x_test, predictions, y_test, epoch, n=self.display_freq)

        # Save the figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        # Display the digits every 'display_freq' number of epochs
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave(GIF_PATH, self.images, fps=1)



my_custom_callback = VisCallback(x_test, y_test)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=28*28),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=64,
          epochs=10,
          callbacks=[my_custom_callback])