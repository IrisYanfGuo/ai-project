#!/usr/local/bin/python3
# File Name: kmeans.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-19 10:34:20
import toolkit
import numpy as np

# 这次不用 np.mat

class kmeans(object):
    """docstring for kmeans"""
    # read trainSet, "attributes", and the number of cluster
    def __init__(self):
        self.__predictions = []
        self.__k = 2
        pass


    def __getEuclideanDistance(self,data1,data2):
        dd = 0.0
        for i in range(len(self.__attributes)):
            dd = dd + np.power(data1[i] - data2[i],2)
        return np.sqrt(dd)

    def __randomCentral(self):
        cluster_lst = []
        for i in range(self.__k):
            cluster_lst.append([])
        for i in range(self.__k):
            index = round(np.random.rand()*len(self.__trainSet))
            cluster_lst[i].append(self.__trainSet[index])
        return cluster_lst


    #print(np.random.rank(k))
    def training(self,attributes,trainSet,k=2):
        self.__k = k
        self.__attributes = attributes

        for i in range(len(trainSet)):
            for j in range(len(attributes)):
                trainSet[i][j] = float(trainSet[i][j])

        self.__trainSet = trainSet
        row,col = len(trainSet)-1,len(attributes)
        cluster_lst = self.__randomCentral()

        for i in range(row):
            minDist = np.inf
            minIndex = 0

            for j in range(k):
                distance = self.__getEuclideanDistance(cluster_lst[j][0],trainSet[i])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            cluster_lst[minIndex].append(trainSet[i])

        self.__predictions = cluster_lst

    def getPrediction(self):
        for i in range(self.__k):
            print("cluster "+str(i)+": ")
            for j in range(len(self.__predictions[i])):
                print(self.__predictions[i][j],end="\t")
            print("\n##################\n")


