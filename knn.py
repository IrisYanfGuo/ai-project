#!/usr/local/bin/python3
# File Name: knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-04-21 21:20:47
import math
from operator import itemgetter
from random import random
from readData import readData


class knn():
    __attributes = []
    __instances = []
    __norm = []
    __trainSet = []
    __testSet = []
    __k = 3
    __split = 0.8

    # read data set
    def __init__(self, filename, split,k=3):
        self.__filename = filename
        self.__k = k
        self.__split = split
        f = readData(self.__filename)
        self.__attributes,self.__instances = f.readDataSet()
        for i in range(len(self.__instances)):
            for j in range(len(self.__attributes)):
                self.__instances[i][j] = float(self.__instances[i][j])


    #divide dataset into training and testing dataset
    def __splitDataSet(self, data, split):
        trainSet = []
        testSet = []

        for i in range(len(data)):
            if random() <= split:
                trainSet.append(data[i])
            else:
                testSet.append(data[i])
        return trainSet, testSet


    #get attributes
    def getAttributes(self):
        return self.__attributes
    #get instances
    def getInstances(self):
        return self.__instances

    def getNormalization(self):
        if len(self.__norm)==0:
            self.__normalization()
        return self.__norm

    def __normalization(self):
        res = []
        for column in range(len(self.__instances)):
            l = len(self.__attributes)
            temp = self.__instances[column][:]

            max_val, min_val = max(temp[:l-1]), min(temp[:l-1])
            #print(min_val)

            if min_val!=max_val:
                t = [float(i)/sum(temp[:l-1]) for i in temp[:l-1]]
                t.append(temp[l])
                res.append(t)
            else:
                res.append(temp)

        return res


    # to calculate the Euclidean distance
    def __getEuclideanDistance(self, data1, data2):
        d = 0.0
        for x in range(len(data1)-1):
            d += pow((data1[x] - data2[x]),2)
        return math.sqrt(d)


    # get K near neighborhoods
    # 获取还有点问题, 字典会导致数据缺失
    def __getKNearNeighbors(self, trainSet, testInstance, n=3):
        distances = []
        neighbors = []
        neighbors2 = []
        dis = {}

        for i in range(len(trainSet)):
            dist = self.__getEuclideanDistance(testInstance,trainSet[i])
            distances.append((trainSet[i],dist))

            xxx = trainSet[i][:];xxx.append(i)
            dis.setdefault(tuple(xxx),dist)

        distances.sort(key=itemgetter(1))

        dis = sorted(dis.items(),key=itemgetter(1))

        for i in range(n):
            neighbors.append(distances[i][0])
            ttt = list(dis[i][0])
            neighbors2.append(ttt[:len(ttt)-1])

            if (neighbors!=neighbors2):
                print(neighbors,end=",")
                print(distances[i][1])
                print(len(distances))
                print(neighbors2,end=",")
                print(dis[i][1])
                print(len(dis))
                print()
        #return n2  #使用 neighbors最好
        return neighbors2


    def __getClassification(self, neighbors):
        dic = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]   # result
            dic.setdefault(response,dic.get(response,0)+1)
        temp = sorted(dic.items(),key=itemgetter(1))
        temp.reverse()
        #最近的几个instances 来得到最大的结果, 值最大,说明靠的近,选大的
        return temp[0][0]


    def __getResult(self, testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        #print(correct)
        return (correct/float(len(testSet))) *100.0


    def training(self):
        predictions = []
        #self.__trainSet,self.__testSet = self.__splitDataSet(self.__instances,self.__split)
        self.__trainSet,self.__testSet = self.__splitDataSet(self.__normalization(),self.__split)

        for i in range(len(self.__testSet)):
            neighbors = self.__getKNearNeighbors(self.__trainSet,self.__testSet[i],self.__k)
            result = self.__getClassification(neighbors)
            predictions.append(result)
        #print(result)
        return self.__getResult(self.__testSet,predictions)
