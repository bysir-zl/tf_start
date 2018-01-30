import os
import tensorflow as tf

# 防止报错
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 声明tf变量
# 由于tf内部是由c++运行, 所以当然py的变量是不行的
W = tf.Variable(0.1, dtype=tf.float32)
b = tf.Variable(-0.1, dtype=tf.float32)
# 声明tf占位符, 类似算式中的变量, 但比`变量`这个概念更抽象
x = tf.placeholder(tf.float32)
# 得到一个线性模型, 也就是算式 y=W*x+b, 可以得到x与y的关系
linear_model = W * x + b
# 开启会话, 也就是开启一个`图`, 他是c++与py之间数据转换的桥梁, c++运行的就是这个

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 需要显式的初始化变量
init = tf.global_variables_initializer()
# 在图上运行这个初始化变量方法
sess.run(init)

# 运行一下这个图, 这里的x是数组, 所以得到结果也是一个数组
print(sess.run(linear_model, {x: [1, 2, 3]}))
# 结果是[0.         0.1        0.20000002]
# 0.1*1+(-0.1) = 0
# 0.1*2+(-0.1) = 0.1
# 0.1*3+(-0.1) = 0.2
# 浮点运算有精度丢失问题, 不管啦

# 接下来需要的是给出x与y, 得到W与b, 读过初中的人都应该会, 但无疑用机器更省心, 这就是机器学习的好处
# 其实机器学习就是给定原数据(x)和期待结果(y) 让机器找出他们两之间的关系, 再给定其他x值的时候就能准确得到y, 这个`找`的动作就是`训练`, 也叫`拟合`
# 乱找肯定是找不到的, 这时就需要`模型`, 模型是描述x和y之间大致关系的算式;
# 比如我们调查路人心目中`老人`定义的`age`值, 根据常识, 老人与age是正比关系,所以我们定义模型为 y = x > b, 我们给定几组xy, 机器学习就能轻易得到b的值;
# 但是如果我们给定 y = x-a>b 模型, 引入了一个不必要的变量, 机器就很难计算出a,b的值,就算计算出来那也只是猜的, 对以后的数据依然计算不准; 所以选定一个模型对于训练至关重要, 它直接关系到训练效率与准确性.

# start~

# 声明一个占位符y, 也就是我们的期待结果
y = tf.placeholder(tf.float32)
# 一个方差算法
squared_deltas = tf.square(linear_model - y)
# 数组求和, 刚刚得到的squared_deltas是一个数组(根据传入的x而定; 如果你传递的x是数组,那么linear_model也是数组)
loss = tf.reduce_sum(squared_deltas)

# 声明一个优化器, 这个优化器就是用来拟合x与y的
# 模型选好了, 还需要选好拟合函数, 好的拟合函数能更快找到正确数据, 也不会出现过度拟合的情况
# 这里选用GradientDescent(梯度下降)算法, 我数学不好, 不知道更多了..
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 好了 现在这个`图`就算完成了, 运行它 这个optimizer就能自动找到最优解
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, 3, 6, 9]})

print(sess.run([W, b]))
