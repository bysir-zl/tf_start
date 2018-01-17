import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable([[1., 1., 1], [2., 2., 1], [2., 2., 1], [2., 2., 1], [2., 2., 1]])
y = tf.nn.softmax(x)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(y))

y_ = tf.log(y)
print(sess.run(y_))
w = tf.reduce_sum(y_)
print(sess.run(w))
