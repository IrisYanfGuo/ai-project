from readData import readData

import math
import operator
from random import random


# iris, continuous variables
# car, discrete variables
class Naive_bayes(object):
    __attributes = []
    __instances = []
    __trainSet = []
    __testSet = []

    __px={}
    __pc={}
    __pxc={}

    def __init__(self, filename, split):
        self.__filename = filename
        f = readData(self.__filename)
        self.__attributes, self.__instances = f.readDataSet()
        self.__trainSet, self.__testSet = self.splitDataSet(split)

    # divide dataset into training and testing dataset
    def splitDataSet(self, split):
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

    def getAttributes(self):
        return self.__attributes
        # get instances

    def getInstances(self):
        return self.__instances

    def printTestset(self):
        for i in self.__testSet:
            print(i)

    def printTrainset(self):
        for i in self.__trainSet:
            print(i)

    def train(self):
        self.count_pc()
        self.count_px()
        self.count_pcx()

    def predict(self):
        pass

    def count_px(self):
        for i in self.__trainSet[:-1]:
            if i not in self.__px.keys():
                self.__px[i] = 1
            else:
                self.__px[i] += 1

    def count_pc(self):
        for i in self.__trainSet[-1]:
            if i not in self.__pc.keycs():
                self.__pc[i] = 1
            else:
                self.__pc[i] += 1
    def count_pcx(self):
        for i in self.__trainSet:
            if i not in self.__pc.keycs():
                self.__pcx[i] = 1
            else:
                self.__pcx[i] += 1


car_naive = Naive_bayes("./txtfile/car_prepro.txt",0.7)
print(car_naive.getAttributes())





