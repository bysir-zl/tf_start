import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 推荐文章: CNN中的maxpool到底是什么原理？(http://www.techweb.com.cn/network/system/2017-07-13/2556494.shtml)

# 输入是[1,3,3,1]，代表1个图像，大小3x3，1个通道(通道的意思就是图片中RBG. 用一个通道可以表示灰度)
input = tf.constant(
    [  # 1
        [  # 3
            [  # 3
                [1],  # 1
                [2],
                [3]
            ],
            [
                [4],
                [5],
                [6]
            ],
            [
                [7],
                [8],
                [9]
            ]
        ]
    ], dtype=tf.float32)

# 池化的作用:
# 1. invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(尺度)
# 2. 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力

# 理解池化(pooling):
#   池化和卷积类似, 也是有一个滑块, 只是运算是取Max或者Avg或者其他, 这根据选取的池化函数而定,
#   下面只提到了max_pooling, 它是在滑块内选出一个最大值作为输出.

# ksize: 滑块大小 [batch,height,width,channels], 一般不会池化batch和channels(我也没理解池化batch和channels的原理和意义),所以一般都是[1,x,x,1] 这样的形式
# strides: 和卷积一样, 步长
# padding: SAME, 关于padding取值和影响的结果请看同级目录下 `padding.py`
y = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y))

'''
[[[[5.]
   [6.]
   [6.]]
  [[8.]
   [9.]
   [9.]]
  [[8.]
   [9.]
   [9.]]]]
'''
