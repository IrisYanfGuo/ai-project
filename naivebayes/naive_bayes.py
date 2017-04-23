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
        pass

    def gauss_dist(self, mu, sigma, x):

        return












