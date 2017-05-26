#!/usr/local/bin/python3
# File Name: main_id3.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 20:39:33
import toolkit as tk 
from AI_id3 import id3
import numpy as np

attributes,instances = tk.readDataSet("dataset/sunburnt.csv")
attributes,test = tk.readDataSet("dataset/sunburnt.csv")

attributes = (list(attributes))

instances = instances.tolist()
test = test.tolist()


myid3 = id3()
myid3.training(attributes,instances)
acc,pre = myid3.getPrediction(test)
print(acc,pre)

#t = myid3.training()
#tk.createPlot(t)
#print(list(t.keys()))
#print(test)
#print(myid3.getResult(test))



#tk.createPlot(t)
