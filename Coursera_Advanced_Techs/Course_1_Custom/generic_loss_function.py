import tensorflow as tf
from tensorflow.keras.losses import Loss

def my_huber_loss(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = abs(error) <= threshold
    small_error_loss = error ** 2 / 2
    big_error_loss = threshold * (abs(error) - (0.5 * threshold))
    return tf.cast(tf.where(is_small_error, small_error_loss, big_error_loss), dtype=tf.float16)

loss = my_huber_loss(1, 1)

print(loss)

def my_huber_loss_with_threshold(threshold):
    '''Only works in model.compile becasue it expects a function that
    takes in y_true and y_pred'''
    def my_huber_loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = abs(error) <= threshold
        small_error_loss = error ** 2 / 2
        big_error_loss = threshold * (abs(error) - (0.5 * threshold))
        return tf.cast(tf.where(is_small_error, small_error_loss, big_error_loss), dtype=tf.float16)
    return my_huber_loss

class MyHuberLoss(Loss):
    threshold = 1
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = abs(error) <= self.threshold
        small_error_loss = error ** 2 / 2
        big_error_loss = self.threshold * (abs(error) - (0.5 * self.threshold))
        return tf.cast(tf.where(is_small_error, small_error_loss, big_error_loss), dtype=tf.float16)
