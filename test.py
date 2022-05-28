# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/5/27 13:49
# @Function:
import keras

model = keras.models.load_model("test_model")
dict = model.get_config()
print(1)
weight = model.get_layer("dense").get_weights()
print(1)

