#!/usr/local/bin/python3
# File Name: main_knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-20 15:14:30

import toolkit as tk
from AI_knn import knn
import numpy as np

attributes, instences = tk.readDataSet("blood.csv")
#print(instences)
attributes = list(attributes)

instences = tk.Z_ScoreNormalization(instences,len(attributes))
print(instences)
#print(instences)

lst = []
for i in range(5):
	trainSet, testSet = tk.splitDataSet(instences, 0.8)
	myknn = knn(attributes,trainSet.tolist())
	acc,result = myknn.getPrediction(testSet.tolist(),3)
	lst.append(acc)



#tk.draw_bar(lst)

print(lst)

print("mean = ",np.mean(lst),";std = ", np.std(lst))
