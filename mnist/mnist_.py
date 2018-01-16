import os

import tensorflow as tf

from mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。
# 如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。
# softmax: 按概率选择最大的值, 越大选中的概率越大, 具体概率运算请百度
# 张量的乘法: 看起来和 矩阵的乘法一样, 第一个矩阵的列数必须等于第二个矩阵的行数
# 张量的加法: 看起来和矩阵的加法一样, 要求两个矩阵的列相同, 如果行不同会尝试填充到满足运算规则(行列都相同)的形状
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, {x: batch_xs, y_: batch_ys})

print(sess.run(b))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))
