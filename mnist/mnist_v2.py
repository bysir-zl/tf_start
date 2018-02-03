import os

import tensorflow as tf

from mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # tf.nn.conv2d是TensorFlow里面实现卷积的函数
    # 1 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
    #   具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
    # 2 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
    #   具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数(输出通道个数)]，要求类型与参数input相同，
    #   有一个地方需要注意，第三维in_channels，等于input参数的第四维
    # 3 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    # 4 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
    #   当padding=SAME时，输入与输出形状相同
    #   padding是VALID时,输出的宽高为 (图片大小-filter大小)/步长+1
    # 5 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
    # 结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # max pooling是CNN当中的最大值池化操作
    # 1 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    # 2 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 3 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    # 4 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
    # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", [None, 784])

# 现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
# 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
# 卷积核个数:32, height:5 width:5 channel:1
W_conv1 = weight_variable([5, 5, 1, 32])
# 而对于每一个输出通道都有一个对应的偏置量。
b_conv1 = bias_variable([32])

# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
# 调整矩阵维度
# 第1个参数为被调整维度的张量。
# 第2个参数为要调整为的形状。
# 返回一个shape形状的新tensor
# 注意shape里最多有一个维度的值可以填写为-1，表示自动计算此维度。
# 第一位-1是自动计算得 height:28 width:28 channel:1
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
# 这里得到的h_conv1是 [None,28,28,32]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 一般卷积后会接一个池化, 池化操作会舍去一部分特征, 但是不会影响识别, 具体原因可以看这篇文章:
#   CNN中的maxpool到底是什么原理？(http://www.techweb.com.cn/network/system/2017-07-13/2556494.shtml)
# 池化操作一般有两种，一种是Avy Pooling,一种是max Pooling,如下是max pooling
# 步长为2 所以得到的是 [None,14,14,32]
h_pool1 = max_pool_2x2(h_conv1)

# ~第二层卷积~

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# [None,14,14,64]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# [None,7,7,64]
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# ~ 我们最后还是要得到一个[10]的结果 代表0-9的几率 ~
# 所以我们用一个权重矩阵去乘(这和基础版mnist一样)

# 最后一层, 将前两层卷积的结果打扁成[None,7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 矩阵相乘, [None,7*7*64] * [7 * 7 * 64, 1024] => [None,1024]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
# dropout也是一个大杀器, 他能防止过拟合
#  但它为何有效,却众说纷纭 (http://blog.csdn.net/stdcoutzyx/article/details/49022443)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# [None,1024] * [1024,10] => [none,10] 也就是目标矩阵
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# y_ 目标矩阵
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
y_ = tf.placeholder("float", [None, 10])

sess = tf.Session()

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(10):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy,
                                    feed_dict={x: mnist.test.images[:1], y_: mnist.test.labels[:1], keep_prob: 1.0}))
