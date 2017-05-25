from toolkit import *
import numpy as np
import pandas as pd

class LogisticRegression(object):

    def __init__(self,attributes,trainSet,testSet):

        self.__attributes = attributes
        self.__trainSet = trainSet
        self.__testSet = testSet
        self.train(self.__trainSet,0,0)

    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))

    def train(self,trainSet,alpha,step):
        # random assign values to weights
        self.__weight = np.random.rand(len(self.__attributes),1)

        target = [trainSet[i][-1] for  i in range(len(trainSet))]
        X = [trainSet[i][:-1] for i in range(len(trainSet))]

        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i][j] = float(X[i][j])

        f_out = X*self.__weight
        error = [target[i]-f_out[i] for i in range(len(target))]




a = pd.read_csv("../dataset/transfusion.csv")

b = a.as_matrix()
print(b[0][1])