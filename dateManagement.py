# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/27 19:40
# @Function:
import keras.models
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np


class dataManagnment:
    @classmethod
    def devide(self, num):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()  # 使用Keras自带的mnist工具读取数据（第一次需要联网）
        x_train = np.array_split(X_train,num)
        y_train = np.array_split(y_train,num)
        return x_train, y_train, X_test, y_test

if __name__ == "__main__":
    dataManagnment.devide(10)