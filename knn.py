#!/usr/local/bin/python3
# File Name: knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-04-21 21:20:47
import math
import operator
from random import random
from readData import readData


class knn():
    __attributes = []
    __instances = []
    __trainSet = []
    __testSet = []

    # read data set
    def __init__(self, filename, split):
        self.__filename = filename
        f = readData(self.__filename)
        self.__attributes,self.__instances = f.readDataSet()
        self.__trainSet,self.__testSet = self.splitDataSet(split)
        self.getPrediction()

    # divide dataset into training and testing dataset
    def splitDataSet(self,split):
        trainSet = []
        testSet = []
        for x in range(len(self.__instances)):
            for y in range(len(self.__attributes)):
                self.__instances[x][y] = float(self.__instances[x][y])
            if random() <= split:
                trainSet.append(self.__instances[x])
            else:
                testSet.append(self.__instances[x])
        return trainSet, testSet


    #get attributes
    def getAttributes(self):
        return self.__attributes
    #get instances
    def getInstances(self):
        return self.__instances

    # to calculate the Euclidean distance
    def getEuclideanDistance(self, data1, data2):
        distance = 0
        for x in range(len(data1)-1):
            distance += pow((data1[x] - data2[x]),2)
        return math.sqrt(distance)


    def getKNearNeighbors(self, trainSet, testInstance, n):
        distances = []
        neighbors = []
        n2 = []
        dis = {}

        for x in range(len(trainSet)):
            dist = self.getEuclideanDistance(testInstance,trainSet[x])
            distances.append((trainSet[x],dist))
            dis.setdefault(tuple(trainSet[x]),dist)

        distances.sort(key=operator.itemgetter(1))

        temp = sorted(dis.items(),key=operator.itemgetter(1))

        for x in range(n):
            neighbors.append(distances[x][0])
            n2.append(list(temp[x][0]))

            if (neighbors!=n2):
                print(neighbors,end=",")
                print(distances[x][1])
                print(len(distances))
                print(n2,end=",")
                print(temp[x][1])
                print(len(temp))
                print()
        return neighbors



    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]


    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        #print(correct)
        return (correct/float(len(testSet))) *100.0


    def getPrediction(self):

        predictions = []
        k = 3
        for x in range(len(self.__testSet)):
            neighbors = self.getKNearNeighbors(self.__trainSet,self.__testSet[x],k)
            result = self.getResponse(neighbors)
            predictions.append(result)
        #print(result)

        accuracy = self.getAccuracy(self.__testSet,predictions)
        print((accuracy))

k = knn("iris.txt",0.66)

