import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/tf2-preview/inception_v3/classification/4'),
    tf.keras.layers.Activation('softmax')
])

model.build([None, 300, 300, 3])

img = cv2.imread('husky.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300)) / 255.
image = np.expand_dims(img, axis=0)

class_index = 251
num_classes = 1001
expected_output = tf.one_hot([class_index] * image.shape[0], num_classes)

with tf.GradientTape() as tape:
    inputs = tf.cast(image, tf.float16)
    tape.watch(inputs)
    predictions = model(inputs)
    loss = tf.keras.losses.categorical_crossentropy(expected_output, predictions)

gradients = tape.gradient(loss, inputs)

grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)
normalized_tensor = tf.cast(
    255 * (grayscale_tensor - tf.reduce_min(grayscale_tensor)) /
    (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)), tf.uint8
)

normalized_tensor = tf.squeeze(normalized_tensor)

max_pixel = np.unravel_index(np.argmax(grayscale_tensor[0]), grayscale_tensor[0].shape)
min_pixel = np.unravel_index(np.argmin(grayscale_tensor[0]), grayscale_tensor[0].shape)

#plt.figure(figsize=(8, 8))
#plt.axis('off')
#plt.imshow(normalized_tensor, cmap='gray')
#plt.show()

gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
gradient_color = gradient_color / 255.
super_imposed = cv2.addWeighted(img, 0.5, gradient_color, 0.5, 0.0)

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(super_imposed)
plt.show()