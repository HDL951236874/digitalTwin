# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/27 19:40
# @Function:
import random
from dateManagement import *
from user import *
from modelGenerator import *
from server import *


class simulation:
    def __init__(self, number):
        self.user_number = number
        self.seed = random.seed(10)
        self.user_pool = []
        self.server = None
        self.modelGenerator = modelGenerator()
        self.epoch = 2
        pass

    def initialize(self):
        x_train, y_train, x_test, y_test = dataManagnment.devide(self.user_number)
        for n in range(self.user_number):
            self.user_pool.append(user(self.modelGenerator, (x_train[n], y_train[n]), n))
        self.server = server((x_test, y_test), self.modelGenerator)

    def simulate(self):
        self.initialize()
        for _ in range(self.epoch):
            parameterList = []
            for index in self.user_pool:
                parameterList.append(index.run())
            self.server.run(parameterList)


if __name__ == '__main__':
    sim = simulation(2)
    sim.simulate()
