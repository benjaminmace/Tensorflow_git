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

vocab_file = open('vocabulary.obj', 'rb')

vocabulary = pickle.load(vocab_file)

encoder = tfds.features.text.TokenTextEncoder(
    list(vocabulary), oov_token='<UNK>', lowercase=True, tokenizer=tokenizer
)

def my_encoder(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(line):
    split_line = tf.strings.split(line, ",", maxsplit=4)
    label_str = split_line[2]
    review = "sostoken" + split_line[4] + "eostoken"
    label = 1 if label_str == 'pos' else 0

    (encoded_text, label) = tf.py_function(
        my_encoder, inp=[review, label], Tout=(tf.int64, tf.int32),
    )

    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map_fn, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(25000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))

ds_test = ds_test.map(encode_map_fn)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0),
    tf.keras.layers.Embedding(input_dim=len(vocabulary)+2, output_dim=32,),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1),
    metrics=['accuracy']
)

model.fit(ds_train, epochs=15)

model.evaluate(ds_test)