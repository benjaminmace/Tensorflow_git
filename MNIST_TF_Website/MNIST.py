import tensorflow as tf
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.
test_images = test_images / 255.

train_images = np.reshape(train_images,(60000, 28, 28, 1))
test_images = np.reshape(test_images, (10000, 28, 28, 1))

model = tf.keras.Sequential()


model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(2, padding='same'))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(2, padding='same'))

model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3),activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(2, padding='same'))


model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(10))

ES = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=5, restore_best_weights=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=100,
                    validation_data=(test_images, test_labels),
                    callbacks=[ES])