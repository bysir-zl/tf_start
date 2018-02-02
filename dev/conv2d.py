import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# filter是[2,2,1,3]，代表filter的大小是2x2，输入通道通道是1 需要与input通道相同, 输出通道是3。
# 理解filter: 他就是一个2x2的滑窗, 每移动一次就与input点对点乘加一次, 就能得到一个数, 从上到下遍历了input之后, 就能得到一个(图片长宽-filter长宽)/步长+1的输出
#   在padding是SAME的情况下, 会自动填充input使得得到的输出和原始input形状一致
#   对于第四个参数(输出通道) 可以理解为用多个(层)filter去依次遍历, 得到的也是多层(通道)的输出
#   比如下面这个filter, 就是用[[1 1][1 1]]的滑块遍历得到输出的第一层, 然后用[[2 2][2 2]]得到输出的第二层
filter = tf.constant(
    [  # 2
        [  # 2
            [  # 1
                [1, 2, 1]  # 1
            ],
            [
                [1, 2, 1]
            ]
        ],
        [
            [
                [1, 2, 1]
            ],
            [
                [1, 2, 1]
            ]
        ]
    ], dtype=tf.float32)

# 这里的padding是SAME, conv2d会自动填充input的形状 使得结果和输入形状相同
y = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y))

'''
 [[[[12. 24. 12.]
    [16. 32. 16.]
    [ 9. 18.  9.]]
   [[24. 48. 24.]
    [28. 56. 28.]
    [15. 30. 15.]]
   [[15. 30. 15.]
    [17. 34. 17.]
    [ 9. 18.  9.]]]]
'''
