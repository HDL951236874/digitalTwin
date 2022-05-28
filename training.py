# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/26 11:21
# @Function:
import keras.models
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

'''
according to the paper, we use full connection network in this experiment.
'''
tBatchSize = 128
VERBOSE = 1
'''第一步：选择模型'''
model = Sequential() #using sequence model

'''第二步：构建网络层'''
model.add(Dense(500, input_shape=(784,)))  # 输入层，28*28=784 输入层将二维矩阵换成了一维向量输入2
model.add(Activation('relu'))  # 激活函数是tanh 为双曲正切
model.add(Dropout(0.5))  # 采用50%的dropout  随机取一半进行训练

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(10))  # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

'''第三步：网络优化和编译'''
model.compile(loss='categorical_crossentropy', optimizer='adam')  # 使用交叉熵作为loss函数

'''第四步：训练'''

# 数据集获取 mnist 数据集的介绍可以参考 https://blog.csdn.net/simple_the_best/article/details/75267863
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 使用Keras自带的mnist工具读取数据（第一次需要联网）

# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# 这个能生成一个OneHot的10维向量，作为Y_train的一行，这样Y_train就有60000行OneHot作为输出
Y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 整理输出
Y_test = (np.arange(10) == y_test[:, None]).astype(int)  # np.arange(5) = array([0,1,2,3,4])

# 非常重要!!!
X_train = X_train / 256.0
X_test = X_test / 256.0

model.fit(X_train, Y_train, batch_size=tBatchSize, epochs=1,
          shuffle=True, verbose=2, validation_split=0.3)

model.save("test_model")
loss = model.evaluate(X_test, Y_test)
print(loss)