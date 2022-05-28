# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/27 18:59
# @Function:
from modelGenerator import *


class user:
    def __init__(self, modelGenerator, data, id):
        self.model = modelGenerator.generate()
        self.dataset = data
        self.id = id
        self.train_time = 0

    def training(self):
        x_train = self.dataset[0]
        y_train = self.dataset[1]
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        y_train = (np.arange(10) == y_train[:, None]).astype(int)  # 整理输出

        x_train = x_train / 256.0

        self.model.fit(x_train, y_train, batch_size=128, epochs=1,
                       shuffle=True, verbose=2, validation_split=0.3)
        self.model.save("model_cache/user_" + str(self.id) + "/" + "sample" + str(self.train_time))
        self.train_time += 1

    def parameters_extraction(self):
        dict = self.model.get_config()
        parametersList = []
        for layer in range(len(dict['layers'])):
            name = dict['layers'][layer]['config']['name']
            if 'dense' in name and 'input' not in name:
                parametersList.append(self.model.get_layer(name).get_weights())
        return parametersList

    def encoding(self, parametersList):
        return parametersList

    def run(self):
        self.training()
        return self.encoding(self.parameters_extraction())
