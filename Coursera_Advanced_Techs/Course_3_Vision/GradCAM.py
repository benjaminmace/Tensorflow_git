import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tfds.disable_progress_bar()

splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
splits, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split = splits)
(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.
    return image, label


train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_batches = test_examples.map(format_image).batch(1)

def build_model():
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMAGE_SIZE + (3, ),
                                                   weights='imagenet',
                                                   include_top=False)

    output = layers.GlobalAveragePooling2D()(base_model.output)

    output = layers.Dense(2, activation='softmax')(output)

    model = tf.keras.Model(base_model.inputs, output)

    for layer in base_model.layers[:-4]:
        layer.trainable=False

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()
    return model

def get_CAM(processed_image, predicted_label, layer_name='block5_conv3'):
    model_grad = tf.keras.Model(
        [model.inputs],
        [model.get_layer(layer_name).output,
         model.output])

    with tf.GradientTape() as tape:
        conv_output_values, predictions = model_grad(processed_image)
        loss = predictions[:, predicted_label]

    grads_values = tape.gradient(loss, conv_output_values)
    grads_values = K.mean(grads_values, axis=(0, 1, 2))

    conv_output_values = np.squeeze(conv_output_values.nump())
    grads_values = grads_values.numpy()

    for i in range(512):
        conv_output_values[:, :, i] *= grads_values[i]

    heatmap = np.mean(conv_output_values, axis=1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    del model_grad, conv_output_values, grads_values, loss

    return heatmap



model = build_model()

model.fit(train_batches,
          epochs=3,
          validation_data=validation_batches)


outputs = [layer.output for layer in model.layers[1:18]]