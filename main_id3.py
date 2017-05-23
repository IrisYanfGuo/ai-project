#!/usr/local/bin/python3
# File Name: main_id3.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 20:39:33
import toolkit as tk 
from AI_id3 import id3
import numpy as np

attributes,instances = tk.readDataSet("dataset/car.txt")
#att,test = tk.readDataSet("dataset/car.txt")
#print(attributes)
lst = []
for i in range(500):
	trainSet,testSet = tk.splitDataSet(instances,0.66)
	myid3 = id3(attributes,trainSet)
	t = myid3.training()
	myid3.getResult(testSet)
	lst.append(myid3.getAccuracy())

print("mean: ",np.mean(lst),"std: ",np.std(lst))

#tk.createPlot(t)
