#!/usr/local/bin/python3
# File Name: knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-04-21 21:20:47
import numpy as np
from operator import itemgetter
import toolkit as tk 

# no np.mat in this program

class knn():
    __attributes = []
    __trainSet = []

    # read trainSet, testSet and "attributes" 
    def __init__(self):
        pass

    # to calculate the Euclidean distance
    def __getEuclideanDistance(self, data1, data2):
        d = 0.0
        for x in range(len(data1)-1):
            d += pow((data1[x] - data2[x]),2)
        return np.sqrt(d)

    # get K near neighborhoods
    # 获取没有问题了, 字典会导致数据缺失
    def __getKNearNeighbors(self, trainSet, testInstance, n=3):
        distances = []
        neighbors = []
        #neighbors2 = []
        #dis = {}

        for i in range(len(trainSet)):
            dist = self.__getEuclideanDistance(testInstance,trainSet[i])
            distances.append((trainSet[i],dist))

        distances.sort(key=itemgetter(1))

        for i in range(n):
            neighbors.append(distances[i][0])
        return neighbors


    def __getClassification(self, neighbors):
        dic = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]   # result
            dic.setdefault(response,dic.get(response,0)+1)
        temp = sorted(dic.items(),key=itemgetter(1))
        temp.reverse()
        #最近的几个instances 来得到最大的结果, 值最大,说明靠的近,选大的
        return temp[0][0]


    def training(self,attributes,trainSet):
        self.__attributes = attributes
        self.__trainSet = trainSet

        # this is for double check, because we need numeric
        for i in range(len(self.__trainSet)):
            for j in range(len(self.__attributes)):
                self.__trainSet[i][j] = float(self.__trainSet[i][j])

    def getPrediction(self,testSet,k=3):
        for i in range(len(testSet)):
            for j in range(len(self.__attributes)):
                testSet[i][j] = float(testSet[i][j])

        predictions = []
        
        for i in range(len(testSet)):
            neighbors = self.__getKNearNeighbors(self.__trainSet,testSet[i],k)
            result = self.__getClassification(neighbors)
            predictions.append(result)


        if (len(testSet[0]) == len(self.__trainSet)):
            Accuracy = np.nan
        else:
            correct = 0
            for i in range(len(testSet)):
                if testSet[i][-1] == predictions[i]:
                    correct += 1
            Accuracy = correct/float(len(testSet)) *100.0

        return Accuracy,predictions
##########################################################

    def getResult(self):
        return self.__predictions

    def getAccuracy(self):
        if (len(self.__testSet[0])<=len(self.__attributes)):
            return ("getAccuracy() ERROR, len(testSet) < len(trainSet)")

        correct = 0
        for i in range(len(self.__testSet)):
            if self.__testSet[i][-1] == self.__predictions[i]:
                correct += 1
        #print(correct)
        return (correct/float(len(self.__testSet))) *100.0

