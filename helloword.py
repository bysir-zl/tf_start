import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
x = sess.run(hello)
print(x)
a = tf.constant(10)
b = tf.constant(32)
x = sess.run(a + b)
print(x)
sess.close()