import math
import operator
from random import random
from numpy import mean
from numpy import var
import toolkit as tk
import pandas as pd


# iris, continuous variables
# car, discrete variables
class Naive_bayes_conti(object):
    def __init__(self, trainSet, attributes, testSet):

        self.__attributes = attributes

        self.__trainSet = trainSet
        self.__testSet = testSet

        self.__pc = {}
        self.__px = []
        self.__pxc = {}

        self.train()
        self.getPrediction()

    # divide dataset into training and testing dataset
    def splitDataSet(self, split):
        trainSet = []
        testSet = []
        for i in range(len(self.__instances)):
            for j in range(len(self.__attributes)):
                self.__instances[i][j] = float(self.__instances[i][j])
        for x in range(len(self.__instances)):
            if random() <= split:
                trainSet.append(self.__instances[x])
            else:
                testSet.append(self.__instances[x])
        return trainSet, testSet

    def getAttributes(self):
        return self.__attributes
        # get instances

    def count(self, list):
        dict = {}

        dict['mu'] = mean(list)
        dict['sigma'] = var(list)

        return dict

    def count_px(self):
        for i in self.__attributes[:-1]:
            temp = self.__trainSet[i]
            tdict = self.count(temp)
            self.__px.append(tdict)

    def count2(self, list):
        dict = {}
        for i in list:
            if i not in dict.keys():
                dict[i] = 1
            else:
                dict[i] += 1
        for i in dict.keys():
            dict[i] = dict[i] / len(list)
        return dict

    def count_pc(self):
        tag = self.__trainSet[self.__attributes[-1]]
        self.__pc = self.count2(tag)

    def count_group(self, b):
        dict = {}
        for i in b.index:
            dict_t = {}
            for j in range(len(b.columns)):
                dict_t[j] = b.loc[i][j]
            dict[i] = dict_t
        return dict

    def count_pxc(self):
        group = self.__trainSet.groupby(self.__attributes[-1])
        group_mean = group.mean()
        group_var = group.var()
        self.__pxc['mu'] = self.count_group(group_mean)
        self.__pxc['sigma'] = self.count_group(group_var)

    def train(self):
        self.count_px()
        self.count_pc()
        self.count_pxc()

    def _nor_distri(self, mu, sigma, x):

        return 1 / (math.sqrt(2 * math.pi) * sigma) * pow(math.e, -(x - mu) ** 2 / (2 * sigma ** 2))

    def predict(self, blist):
        result = {}
        for i in self.__pc.keys():
            pxc = 1
            px = 1
            for j in range(len(self.__attributes) - 1):
                pxc *= self._nor_distri(self.__pxc['mu'][i][j], self.__pxc['sigma'][i][j], blist[j])
                px *= self._nor_distri(self.__px[j]['mu'], self.__px[j]['sigma'], blist[j])

            result[i] = pxc * self.__pc[i] / px

        largest = max(result.values())

        for i in self.__pc.keys():
            if result[i] == largest:
                t = i
        return t

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet.iloc[i][-1] == predictions[i]:
                correct += 1
        # print(correct)
        return (correct / float(len(testSet))) * 100.0

    def getPrediction(self):
        predictions = []
        for i in range(len(self.__testSet)):
            result = self.predict(self.__testSet.iloc[i])
            predictions.append(result)

        # print(result)
        accuracy = self.getAccuracy(self.__testSet, predictions)
        print(accuracy)


attri, iris = tk.readCsv("../dataset/transfusion.csv")
iris = iris.fillna(method='pad')
print(iris)
iris_naive = Naive_bayes_conti(iris, attri, iris)
c = iris.groupby([attri[-1]]).var()


def count_group(b):
    dict = {}
    for i in b.index:
        dict_t = {}
        for j in range(len(b.columns)):
            dict_t[j] = b.loc[i][j]
        dict[i] = dict_t
    return dict


print(count_group(c))
