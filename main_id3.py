#!/usr/local/bin/python3
# File Name: main_id3.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 20:39:33
import toolkit as tk 
from AI_id3 import id3
import numpy as np

attributes,instances = tk.readDataSet("dataset/car.txt")
att,test = tk.readDataSet("dataset/car.txt")
#print(attributes)
myid3 = id3(attributes,instances)
t = myid3.training()
#tk.createPlot(t)
#print(list(t.keys()))
#print(test)
print(myid3.getResult(test))



#tk.createPlot(t)