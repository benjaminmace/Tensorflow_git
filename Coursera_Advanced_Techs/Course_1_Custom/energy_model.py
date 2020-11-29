import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



scaler = preprocessing.StandardScaler()
dataset = scaler.fit_transform(pd.read_excel('ENB2012_data.xlsx').to_numpy())

features = dataset[:, :-2]
y1 = dataset[:, -2]
y2 = dataset[:, -1]

def scheduler_2025(epoch, lr):
    if epoch < 2025:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def scheduler_200(epoch, lr):
    if epoch % 200 == 0:
        return lr * tf.math.exp(-0.1)
    else:
        return lr

callback_2025 = tf.keras.callbacks.LearningRateScheduler(scheduler_2025)
callback_200 = tf.keras.callbacks.LearningRateScheduler(scheduler_200)

input_layer = Input(8)
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

y1_output = Dense(units='1', name='y1_output')(second_dense)

third_dense = Dense(units='64', activation='relu')(second_dense)

y2_output = Dense(units='1', name='y2_output')(third_dense)

model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss={'y1_output': 'mse', 'y2_output':'mse'},
              metrics={'y1_output':tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output':tf.keras.metrics.RootMeanSquaredError()})

model.fit(x=features, y=[y1, y2],
          batch_size=len(dataset),
          epochs=1,
          validation_split=0.2,
          shuffle=True,
          callbacks=[callback_2025])

