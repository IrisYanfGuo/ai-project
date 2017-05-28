#!/usr/local/bin/python3
# File Name: main_kmeans.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-21 00:22:24

import toolkit as tk
import numpy as np
from AI_kmeans import kmeans
import matplotlib.pyplot as plt

attributes, instences = tk.readDataSet("iris.csv")
attributes = attributes.tolist()
instences2 = instences[:, :len(attributes)]
# instences2 = tk.zero_one_Normalization(instences2,len(attributes))
# instences2 = tk.Z_ScoreNormalization(instences2,len(attributes))

k = 3

mykmeans = kmeans(attributes, instences2)
cp, ct = mykmeans.getPrediction(k)

numSamples, dim = np.shape(instences2)

print("Final cluster centroids: ")

for i in range(k):
    min_dist = np.inf
    result = ""
    count = 0
    for j in range(np.shape(instences)[0]):
        dist = np.sqrt(np.sum(np.power(cp[i, :] - instences2[j, :], 2)))
        if min_dist > dist:
            min_dist = dist
            result = instences[j, np.shape(instences)[1] - 1]
        if (ct[j, 1] == i):
            count += 1
    print(i, end=" ")
    print(cp[i, :], end=" ")
    print(count, end=" ")
    print(result)

print("SSE: ", end="")
E = 0.0
for i in range(k):
    for j in range(np.shape(instences2)[0]):
        if i == ct[j, 1]:
            # print(ct[j,0])
            E += np.sum(np.power(instences2[int(ct[j, 0]), :] - cp[i, :], 2))
print(E)
'''
for j in range(k):
    print("cluster: ",j,end=" --> ")
    print(cp[j,:])
    for i in range(np.shape(instences)[0]):
        index = int(ct[i,1])
        if index==j:
            print(instences[i,:])
    print("\n##################\n")
'''

if (dim == 2):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # draw all samples
    for i in range(numSamples):
        markIndex = int(ct[i, 1])
        plt.plot(instences[i, 0], instences[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(cp[i, 0], cp[i, 1], mark[i], markersize=12)

    plt.show()
