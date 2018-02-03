import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 原文: http://blog.csdn.net/rain6789/article/details/78754516
# 本文在上文基础上加上了自己的理解与代码

# 在CNN中，卷积和池化是一种很常见的操作，一般认为通过卷积和池化可以降低输入图像的维度，也可以达到一定的旋转不变性和平移不变性；
# 而在这种操作过程中，图像(或者特征图)的尺寸是怎么变化的呢？知道卷积核池化之后的大小
# 本文主要描述TensorFlow中，使用不同方式做填充后(padding = 'SAME' or 'VALID' )的tensor的size变化。

# 输入是[1,3,3,1]，代表1个图像，大小3x3，1个通道(通道的意思就是图片中RBG. 用一个通道可以表示灰度)
input = tf.constant(0.1, shape=[1, 28, 28, 1])

# padding: SAME, 结果shape尺寸计算如下:
#   out_size = ceil(in_size / strides)
#   为了得到这个尺寸, 函数会在max_pool之前向input填充必要的0
# 比如这个 ceil(28/2) = 14
y = tf.nn.max_pool(input, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')
print(y.shape)
'''
(1, 14, 14, 1)
'''

# padding: VALID, 结果shape尺寸计算如下:
#   out_size = ceil( (in_size - kernel_size + 1) / strides)
# 比如这个 ceil((28-5+1)/2) = 12
y = tf.nn.max_pool(input, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='VALID')
print(y.shape)
'''
(1, 12, 12, 1)
'''
