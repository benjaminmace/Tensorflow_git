import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.experimental.set_policy(policy)


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train / 255.
X_test = X_test / 255.

def show_img(img):
    img = np.array(img, dtype='float')
    img = img.reshape((28, 28))
    plt.imshow(img)
    plt.show()


def show_cam(image_index):
    features_for_img = features[image_index, :, :, :]
    prediction = np.argmax(results[image_index])
    class_activation_weights = gap_weights[:, prediction]
    class_activation_features = sp.ndimage.zoom(features_for_img, (28/3, 28/3, 1), order=2)
    cam_output = np.dot(class_activation_features, class_activation_weights)


model = tf.keras.Sequential()

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=256,  kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=512,  kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.LayerNormalization(axis=1, center=True, scale=True))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(lr=2e-5)
opt = mixed_precision.LossScaleOptimizer(optimizer)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

SCH = tf.keras.callbacks.LearningRateScheduler(scheduler)



ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=5)

model.fit(X_train,
          y_train,
          batch_size=32,
          epochs=100,
          validation_split=0.2,
          shuffle=True,
          callbacks=[SCH, ES])