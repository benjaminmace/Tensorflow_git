import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

input = Input(shape=(28, 28))

x = Flatten()(input)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

func_model = Model(inputs=input, outputs=predictions)



