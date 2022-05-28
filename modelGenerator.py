# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/27 19:00
# @Function:
import keras.models
from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import yaml


class modelGenerator:
    def __init__(self):
        self.model = None
        self.configPath = "dlModel.yml"

    def readConfig(self):
        with open(self.configPath, "r") as f:
            temp = yaml.load(f.read())
        return temp

    def generate(self):
        model = keras.Sequential()
        config = self.readConfig()
        for n in range(config['num']):
            layer = config['structure_' + str(n+1)]
            if layer['class'] == 'Dense':
                if 'input_shape' in layer:
                    model.add(Dense(int(layer['deep']), input_shape=(int(layer['input_shape']), )))
                else:
                    model.add(Dense(int(layer['deep'])))
            if layer['class'] == 'Activation':
                model.add(Activation(layer['function']))
            if layer['class'] == 'Dropout':
                model.add(Dropout(float(layer['parameter'])))
        model.compile(loss='categorical_crossentropy', optimizer='adam')  # 使用交叉熵作为loss函数

        return model


if __name__ == '__main__':
    model = modelGenerator()
    dlModel = model.generate()
    print(1)
