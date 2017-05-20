#!/usr/local/bin/python3
# File Name: knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-04-21 21:20:47
import numpy as np
from operator import itemgetter

# no np.mat in this program

class knn():
    __attributes = []
    __trainSet = []
    __testSet = []
    __predictions = []
    __k = 3

    # read trainSet, testSet and "attributes" 
    def __init__(self,attributes,trainSet,testSet,k=3):
        self.__k = k
        self.__attributes = attributes
        self.__trainSet = trainSet
        self.__testSet = testSet

        # this is for double check, because we need numeric
        for i in range(len(self.__trainSet)):
            for j in range(len(self.__attributes)):
                self.__trainSet[i][j] = float(self.__trainSet[i][j])

        for i in range(len(self.__testSet)):
            for j in range(len(self.__attributes)):
                self.__testSet[i][j] = float(self.__testSet[i][j])


    # LaTex：{x}_{normalization}=\frac{x-Min}{Max-Min}   min~max Normalization
    def zero_one_normalization(self,data):
        res = data[:]

        nrow = len(data)
        ncol = len(self.__attributes) #to get the columns,

        for i in range(ncol): #each columns
            maxx = max(data[x][i] for x in range(nrow))  #max in each columns
            minx = min(data[x][i] for x in range(nrow))

            if maxx - minx != 1.0:
                for j in range(nrow): #row
                    res[j][i] = (data[j][i] - minx)/(maxx - minx)

        return res

    # LaTex：{x}_{normalization}=\frac{x-\mu }{\sigma }    sigma 标注话
    def Z_ScoreNormalization(self,data):
        res = data[:]

        nrow = len(data)
        ncol = len(self.__attributes)

        for i in range(ncol): #竖排
            average = np.average([data[x][i] for x in range(nrow)])
            sigma = np.std([data[x][i] for x in range(nrow)])
            #print(average,sig)

            if sigma != 0:
                for j in range(nrow): #横排
                    res[j][i] = (data[j][i] - average) / sigma

        return res

    def log_Normalization(self,data):
        res = data[:]

        nrow = len(data)
        ncol = len(self.__attributes)

        for i in range(ncol):
            logg = max(data[x][i] for x in range(nrow))
            if np.log10(logg) !=0:
                for j in range(nrow): #横排
                    res[j][i] = np.log10(data[j][i]) / np.log10(logg)

        return res

    def atan_Normalization(self,data):
        res = data[:]

        nrow = len(data)
        ncol = len(self.__attributes)

        for i in range(ncol):
            for j in range(nrow): #横排
                res[j][i] = np.arctan(data[j][i])*2 / np.pi

        return res

    # to calculate the Euclidean distance
    def __getEuclideanDistance(self, data1, data2):
        d = 0.0
        for x in range(len(data1)-1):
            d += pow((data1[x] - data2[x]),2)
        return np.sqrt(d)

    # get K near neighborhoods
    # 获取没有问题了, 字典会导致数据缺失, list不会
    def __getKNearNeighbors(self, trainSet, testInstance, n=3):
        distances = []
        neighbors = []
        neighbors2 = []

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


    def training(self):
        predictions = []
        self.__trainSet = self.zero_one_normalization(self.__trainSet)
        self.__testSet = self.zero_one_normalization(self.__testSet)

        self.__trainSet = self.atan_Normalization(self.__trainSet)
        self.__testSet = self.atan_Normalization(self.__testSet)

        for i in range(len(self.__testSet)):
            neighbors = self.__getKNearNeighbors(self.__trainSet,self.__testSet[i],self.__k)
            result = self.__getClassification(neighbors)
            predictions.append(result)
        self.__predictions = predictions

    def getResult(self):
        return self.__predictions

    def getAccuracy(self):
        correct = 0
        for i in range(len(self.__testSet)):
            if self.__testSet[i][-1] == self.__predictions[i]:
                correct += 1
        #print(correct)
        return (correct/float(len(self.__testSet))) *100.0



