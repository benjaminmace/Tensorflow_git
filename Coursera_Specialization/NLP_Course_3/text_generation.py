import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


song = """When I get older losing my hair\nMany years from now\nWill you still be sending me a valentine\nBirthday greetings, bottle of wine?\nIf I'd been out till quarter to three\nWould you lock the door?\nWill you still need me, will you still feed me\nWhen I'm sixty four?\nYou'll be older too\nAnd if you say the word\nI could stay with you\n
I could be handy, mending a fuse\nWhen your lights have gone\nYou can knit a sweater by the fireside\nSunday mornings go for a ride\nDoing the garden, digging the weeds\nWho could ask for more?\nWill you still need me, will you still feed me\nWhen I'm sixty four?\n
Every summer we can rent a cottage in the Isle of Wight\nIf it's not too dear\nWe shall scrimp and save\nGrandchildren on your knee\nVera, Chuck and Dave\nSend me a postcard, drop me a line\nStating point of view\nIndicate precisely what you mean to say\nYours sincerely, wasting away\nGive me your answer, fill in a form\nMine forevermore\n
Will you still need me, will you still feed me\nWhen I'm sixty four?\nHo!"""

corpus = song.lower().split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs = input_sequences[:, :-1]
ys = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
history = model.fit(xs, ys, epochs=150, verbose=0)

seed_text = "When I'm sixty"

next_word = 15

for _ in range(next_word):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0))
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' ' + output_word

print(seed_text)

