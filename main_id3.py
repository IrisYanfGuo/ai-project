#!/usr/local/bin/python3
# File Name: main_id3.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 20:39:33
import toolkit as tk 
from AI_id3 import id3
import numpy as np

attributes,instances = tk.readDataSet("sunburnt.csv")
attributes,test = tk.readDataSet("sunburnt.csv")

attributes = (list(attributes))
#print(attributes)

instances = instances.tolist()
test = test.tolist()


myid3 = id3(attributes,instances)
acc,pre = myid3.getPrediction(test)
print(acc,pre)


tk.createPlot(myid3.getTree())
