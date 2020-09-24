import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
import pickle

def filter_train(line):
    split_line = tf.strings.split(line, ",", maxsplit=4)
    dataset_belonging = split_line[1]
    sentiment_category = split_line[2]

    return (
        True
        if dataset_belonging == 'train' and sentiment_category != 'unsup'
        else False
    )

def filter_test(line):
    split_line = tf.strings.split(line, ",", maxsplit=4)
    dataset_belonging = split_line[1]
    sentiment_category = split_line[2]

    return (
        True
        if dataset_belonging == 'test' and sentiment_category != 'unsup'
        else False
    )

ds_train = tf.data.TextLineDataset('Datasets/imdb.csv').filter(filter_train)
ds_test = tf.data.TextLineDataset('Datasets/imdb.csv').filter(filter_test)

tokenizer = tfds.features.text.Tokenizer()

def build_vocabulary(ds_train, threshold=200):
    frequences = {}
    vocabulary = set()
    vocabulary.update(['sostoken'])
    vocabulary.update(['eostoken'])

    for line in ds_train.skip(1):
        split_line = tf.strings.split(line, ",", maxsplit=4)
        review = split_line[4]
        tokenizer_text = tokenizer.tokenize(review.numpy().lower())

        for word in tokenizer_text:
            if word not in frequences:
                frequences[word] = 1
            else:
                frequences[word] += 1

            if frequences[word] == threshold:
                vocabulary.update(tokenizer_text)

    return vocabulary

vocabulary = build_vocabulary(ds_train)
vocab_file = open('vocabulary.obj', 'wb')
pickle.dump(vocabulary, vocab_file)