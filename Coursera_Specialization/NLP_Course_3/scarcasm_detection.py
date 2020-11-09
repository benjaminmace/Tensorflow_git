import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 32
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = '<OOV>'
TRAINING_SIZE = 20000

sentences = []
labels = []

with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = np.array(sentences[0:TRAINING_SIZE])
testing_sentences = np.array(sentences[TRAINING_SIZE:])

training_labels = np.array(labels[0:TRAINING_SIZE])
testing_labels = np.array(labels[TRAINING_SIZE:])

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,
                                maxlen=MAX_LENGTH,
                                padding=PADDING_TYPE,
                                truncating=TRUNC_TYPE)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,
                               maxlen=MAX_LENGTH,
                               padding=PADDING_TYPE,
                               truncating=TRUNC_TYPE)


model = tf.keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    layers.GlobalAveragePooling1D(),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

print(model.summary())


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


history = model.fit(training_padded,
                    training_labels,
                    epochs=30,
                    validation_data=(testing_padded, testing_labels))
