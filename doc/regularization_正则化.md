正则化(regularization)用来解决过拟合问题

## 参考 
1. http://www.cnblogs.com/ooon/p/4964441.html

## 什么是过拟合
看本目录下 overfit_过拟合.md

> 过拟合表现在训练数据上的误差非常小，而在测试数据上误差反而增大

## 如何解决过拟合

方法之一就是正则化(Regularization):

保持所有特征，但是减小每个特征的权重大小，使其对分类y所做的贡献很小
