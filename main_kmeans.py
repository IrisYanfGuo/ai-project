#!/usr/local/bin/python3
# File Name: main_kmeans.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 00:22:24

import toolkit as tk 
import numpy as np 
from AI_kmeans import kmeans
import matplotlib.pyplot as plt


attributes,instences = tk.readDataSet("iris.csv")

#print(instences)
instences = instences[:,:2]
k = 4
mykmeans = kmeans(attributes.tolist(),instences,k)
cp,ct = mykmeans.getPrediction()


numSamples, dim = np.shape(instences)

for j in range(k):
    print("cluster: ",j,end=" --> ")
    print(cp[j,:])
    for i in range(np.shape(instences)[0]):
        index = int(ct[i,1])
        if index==j:
            print(instences[i,:])
    print("\n##################\n")

if (dim==2):
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
	# draw all samples  
	for i in range(numSamples):
		markIndex = int(ct[i, 1])
		plt.plot(instences[i, 0], instences[i, 1], mark[markIndex])  

	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
	# draw the centroids  
	for i in range(k):
		plt.plot(cp[i, 0], cp[i, 1], mark[i], markersize = 12)  

	plt.show()  