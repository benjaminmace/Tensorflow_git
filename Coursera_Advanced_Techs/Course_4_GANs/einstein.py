import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from pprint import pprint

x = tf.constant([1, 2, 3, 4], shape=(2, 2))

xT = tf.transpose(x)

pprint(x)
pprint(xT)

G = tf.linalg.einsum('ij,ij->ij', x, x)

#pprint(G)

G2 = tf.linalg.einsum('ij,jk->ik', xT, x)
#pprint(G2)

G3 = tf.linalg.einsum('ij,jk->', xT, x)
pprint(G3)