import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import random
import numpy as np

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images = train_images / 255.0
test_images = test_images / 255.0


def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


def create_pairs_on_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')

    return pairs, y


def initialize_base_network():
    input = Input(shape=(28, 28,))
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=input, outputs=x)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss


# alternative implementation of contrastive loss

class ContrastiveLoss(Loss):
    margin = 0

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)

ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)

base_network = initialize_base_network()

input_a = Input(shape=(28, 28,))
input_b = Input(shape=(28, 28,))

vect_output_a = base_network(input_a)
vect_output_b = base_network(input_b)

output = tf.keras.layers.Lambda(euclidean_distance,
                                output_shape=eucl_dist_output)([vect_output_a, vect_output_b])

model = Model([input_a, input_b], output)

model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=tf.keras.optimizers.RMSprop())

history = model.fit([tr_pairs[:, 0],
                     tr_pairs[:, 1]],
                    tr_y,
                    epochs=20,
                    batch_size=128,
                    validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y))


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


loss = model.evaluate(x=[ts_pairs[:, 0], ts_pairs[:, 1]], y=ts_y)

y_pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
train_accuracy = compute_accuracy(tr_y, y_pred_train)

y_pred_test = model.predict([ts_pairs[:, 0], ts_pairs[:, 1]])
test_accuracy = compute_accuracy(ts_y, y_pred_test)

print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
