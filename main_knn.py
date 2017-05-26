#!/usr/local/bin/python3
# File Name: main_knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-20 15:14:30

import toolkit as tk
from AI_knn import knn
import numpy as np

attributes, instences = tk.readDataSet("dataset/iris.csv")
attributes = list(attributes)

instences = instences.tolist()


lst = []
for i in range(5):
    trainSet, testSet = tk.splitDataSet(instences, 0.8)
    myknn = knn()
    myknn.training(attributes,trainSet)
    acc,result = myknn.getPrediction(testSet,3)
    lst.append(acc)



tk.draw_bar(lst)

# print(lst)

# print("mean = ",np.mean(lst),";std = ", np.std(lst))
