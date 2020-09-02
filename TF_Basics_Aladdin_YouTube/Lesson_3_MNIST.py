import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

#Sequential API (very convenient, not very flexible)

#model = keras.Sequential(
#    [
#        keras.Input(shape=28*28),
#        layers.Dense(512, activation='relu'),
#        layers.Dense(256, activation='relu', name='my_layer'),
#        layers.Dense(10)
#    ]
#)

#model = keras.Model(inputs=model.inputs,
#                    outputs=[model.layers[-2].output])

#model = keras.Model(inputs=model.inputs,
#                    outputs=[model.get_layer('my_layer').output])

#model = keras.Model(inputs=model.inputs,
#                    outputs = [layer.output for layer in model.layers])

#feature = model.predict(x_train)
#print(feature.shape)

#features = model.predict(x_train)

#for feature in features:
#    print(feature.shape)



#Functional API
inputs = keras.Input(shape=(784), name='input_layer')
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dropout(0.1, name='first_dropout')(x)
x = layers.Dense(512, activation='relu', name='third_layer')(x)
x = layers.Dense(256, activation='relu', name='fourth_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=7, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)