import os

import tensorflow as tf

from mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("data/", one_hot=True)

batch_size = 100
batch_num = mnist.train.num_examples // batch_size
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
# 知识点: 这里的None表示此张量的第一个维度可以是任何长度的。
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b = tf.Variable(tf.zeros([100]) + 0.1)
L = tf.nn.tanh(tf.matmul(x, W) + b)

W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L, W2) + b2)

# 为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。
# 如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。

# 我们用x[None, 784]与一个矩阵W[784, 10]相乘, 就能得到一个[None, 10]的矩阵, 这个矩阵就能表示这[None, 784]中的每张图片最大概率表示的数是什么, 我们一般称W为权重(Weight)矩阵
# 但每种数字图片有其他因素影响, 可能会造成结果始终不能与每个数字完全匹配, 所以最好在加一个[10]的偏移量
# 不过实验得出 是在这个方案里, 不加b这个偏移量影响也不大

# softmax: 按概率选择最大的值, 越大选中的概率越大, 具体概率运算请百度
# 张量的乘法: 看起来和 矩阵的乘法一样, 第一个矩阵的列数必须等于第二个矩阵的行数
# 张量的加法: 看起来和矩阵的加法一样, 要求两个矩阵的列相同, 如果行不同会尝试填充到满足运算规则(行列都相同)的形状
# 得到的y是一个概率数组:
# [None,10] e.g:
# [[0.1 0.9. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.7 0.3 0.]]
y = tf.nn.softmax(L2)

# y_ 目标矩阵
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
y_ = tf.placeholder("float", [None, 10])

# 如何将y拟合到目标矩阵?
# 直接相乘(不是矩阵乘法), 得到的是同标号各数相乘后的结果
# [[1. 1. 1.]
#  [2. 2. 1.]]
# *
# [[1. 1. 1.]
#  [2. 2. 1.]]
# =
# [[1. 1. 1.]
#  [4. 4. 1.]]

# 二次代价
cross_entropy = tf.reduce_sum(tf.square(y - y_))
# 梯度下降, 使cross_entropy最小, 也就是使y_和y最拟合
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# 把cross添加到观测中
tf.summary.scalar("cross", cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 把accuracy添加到观测中
tf.summary.scalar("accuracy", accuracy)

# 获取所有监测的tensor
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('z://tmp/mnist_logs', sess.graph)

for i in range(10):
    for batch in range(batch_num):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, {x: batch_xs, y_: batch_ys})

        # 观测的tensor也需要run
        summary_str = sess.run(merged, {x: batch_xs, y_: batch_ys})

        # 写入到指定目录里
        # 查看需要借助tensorboard, 命令如下: tensorboard --logdir="z://tmp/mnist_logs"
        summary_writer.add_summary(summary_str, i * batch_num + batch)

    print("test accuracy:", sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))
    print("train accuracy:", sess.run(accuracy, {x: batch_xs, y_: batch_ys}))
    print("=====")

# 计算识别测试集的准确度

# argmax: 最大的那个数值所在的下标
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))

# 计算test里第一个图片的数值
out = tf.argmax(y, 1)
print(sess.run(out, {x: mnist.test.images[:1]}))
