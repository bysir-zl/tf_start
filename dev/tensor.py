import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable([[1., 1., 1], [2., 2., 1], [2., 2., 1], [2., 2., 1], [2., 2., 1]])
w1 = tf.Variable([[1., 1], [2., 2], [2., 2]])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 乘法

# 第一个矩阵的列数必须等于第二个矩阵的行数, 乘积是 第二个矩阵的列数*第一个矩阵的行数的矩阵
# [3*5] * [2*3] = [5*2]
y = tf.matmul(x, w1)
print(sess.run(x))
print("*")
print(sess.run(w1))
print("=")

y_ = tf.matmul(x, w1)
print(sess.run(y_))

# 加法
# 按理说矩阵的加法只能是两个维度一样(行列都一样)的矩阵才能运算, 但在tensorFlow里只要列相同就行, 前提是行只能是一行.
# 猜想是在tf里, 会自动将一行的矩阵填充到能满足运算的形状
print("+")
w2 = tf.Variable([[2., 3.]])
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(w2))

yy = y_ + w2
print("=")

print(sess.run(yy))
