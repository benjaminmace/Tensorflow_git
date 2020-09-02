import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

### My Own Pretrained Model with Updates ###
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

#model = keras.models.load_model('Lesson_11_Complete_Model/')

#base_inputs = model.layers[0].input
#base_outputs = model.layers[-2].output
#x = layers.Dense(64, activation='relu', name='dense1')(base_outputs)
#final_outputs = layers.Dense(10, name='dense2')(x)

#model = keras.Model(inputs=base_inputs, outputs=final_outputs)


### Kears Pretrained Model ###

x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

model = keras.applications.InceptionV3(include_top=True)

base_input = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(5)(base_outputs)

new_model = keras.Model(inputs=base_input, outputs=final_outputs)

new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

new_model.fit(x, y, epochs=15)

## TF HUB Models ##

url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'

base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
base_model.trainable = False

model_hub = keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(5)
])

model_hub.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

model_hub.fit(x, y, epochs=15)