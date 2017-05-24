#!/usr/local/bin/python3
# File Name: main_knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-20 15:14:30

import toolkit as tk
from AI_knn import knn
import numpy as np

attributes, instences = tk.readDataSet("dataset/iris.txt")

# trainSet,testSet = tk.splitDataSet(instences,0.8)
# myknn = knn(attributes,trainSet,testSet,3)
new_testSet = []
for t in testSet:
    new_testSet.append(t[:len(attributes)])
print(new_testSet)

lst = []
for i in range(1):
    trainSet, testSet = tk.splitDataSet(instences, 0.8)
    myknn = knn(attributes, trainSet, testSet, 3)
    myknn.training()
    print(myknn.getResult())
    print(myknn.getAccuracy())



# tk.draw_bar(lst)

# print(lst)

# print("mean = ",np.mean(lst),";std = ", np.std(lst))
