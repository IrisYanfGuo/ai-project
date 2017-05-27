#!/usr/local/bin/python3
# File Name: main_kmeans.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 00:22:24

import toolkit as tk 
import numpy as np 
from AI_kmeans import kmeans
from AI_knn import knn

attributes,instences = tk.readDataSet("iris.csv")
attributes = list(attributes)

#instences =  instences.tolist()
#instences = tk.atan_Normalization(instences,len(attributes))

mykmeans = kmeans(attributes,instences,3)
mykmeans.getPrediction()

