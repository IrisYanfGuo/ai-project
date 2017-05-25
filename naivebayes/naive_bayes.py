import math
import operator
import pandas as pd
from random import random
import toolkit as tk
import time


# iris, continuous variables
# car, discrete variables
class Naive_bayes(object):
    def __init__(self, trainSet, attributes,testSet):

        self.__trainSet = trainSet
        self.__attributes = attributes

        self.__testSet = testSet

        self.__pc = {}
        self.__px = []
        self.__pxc = []

        self.train()
        self.getPrediction()

    # divide dataset into training and testing dataset
    def splitDataSet(self, split):
        trainSet = []
        testSet = []
        for x in range(len(self.__instances)):
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

    def count(self, cat):
        dict = {}
        temp = cat.value_counts()
        key = temp.index
        for i in key:
            dict[i] = temp[i]/len(cat)
        return dict

    def count_cat2(self,cat):
        colname = cat.columns
        dict = {}

        # initialize index
        for i in cat.index.levels[0]:
            dict_t1 = {}
            for j in cat.index.levels[1]:
                dict_t1[j] = 0

            dict[i] = dict_t1

        for i in range(len(cat[colname[-1]])):
            b = cat[colname[-1]].index[i]
            t = cat[colname[-1]][i]
            # 如果不存在,设初值为0.001
            if (math.isnan(t)):
                dict[b[0]][b[1]] = 0.001
            else:
                dict[b[0]][b[1]] = t

        return dict

    def count_px(self):
        for i in self.__attributes[:-1]:
            col_value = self.__trainSet[i]
            t_dict = self.count(col_value)
            self.__px.append(t_dict)

    def count_pc(self):
        self.__pc = self.count(self.__trainSet[self.__attributes[-1]])

    def count_pxc(self):
        for i in self.__attributes[:-1]:
            cat = self.__trainSet.groupby([self.__attributes[-1], i]).count()
            dict_t = self.count_cat2(cat)
            for i in dict_t.keys():
                for j in dict_t[i].keys():
                    dict_t[i][j] = dict_t[i][j]/(self.__pc[i]*len(self.__trainSet))
            self.__pxc.append(dict_t)

    def train(self):
        self.count_pc()
        self.count_px()
        self.count_pxc()





    def predict(self, blist):
        result = {}
        for i in self.__pc.keys():
            pxc = 1
            px = 1
            for j in range(len(self.__attributes)-1):
                pxc *= self.__pxc[j][i][blist[j]]
                px *= self.__px[j][blist[j]]

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


attri, car = tk.readCsv("../dataset/car.csv")

train = car[:1200]
test = car[1201:-1]
#print(car)
#print(test)

car_naive = Naive_bayes(train, attri,train)


#print(car.index)
