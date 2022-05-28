# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/27 18:59
# @Function:
import numpy as np


class server:
    def __init__(self, data, modelGenerator):
        self.dataset = data
        self.model = modelGenerator.generate()

    def decoding(self, paramsList):
        return paramsList

    def update(self, paramsList):
        config = self.model.get_config()
        layers = config['layers']
        layerNameList = []
        for layer in layers:
            if 'dense' in layer['config']['name'] and 'input' not in layer['config']['name']:
                layerNameList.append(layer['config']['name'])

        newParamsList = []
        for n in range(len(paramsList[0])):
            now = paramsList[0][n]
            for m in range(len(paramsList)):
                now = self.sum(now, paramsList[m][n])
            newParamsList.append(self.divide(now, len(paramsList)))

        for n in range(len(layerNameList)):
            self.model.get_layer(layerNameList[n]).set_weights(newParamsList[n])

        print(1)

    def testing(self):
        x_test = self.dataset[0]
        y_test = self.dataset[1]
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
        y_test = (np.arange(10) == y_test[:, None]).astype(int)  # 整理输出

        x_test = x_test / 256.0

        score = self.model.evaluate(x_test, y_test, verbose=1)
        print(score)

    def sum(self, paramsListA, paramsListB):
        newList = []
        for n in range(len(paramsListA)):
            newList.append(paramsListA[n] + paramsListB[n])
        return newList

    def divide(self, paramsList, num):
        newList = []
        for index in paramsList:
            newList.append(index / num)
        return newList

    def run(self, paramsList):
        paramsList = self.decoding(paramsList)
        self.update(paramsList)
        self.testing()
