from toolkit import *
import numpy as np
import pandas as pd
import toolkit as tk

class LogisticRegression(object):

    def __init__(self,attributes,trainSet,testSet,alpha=0.1,step =20):

        self.__attributes = attributes
        self.__trainSet = trainSet
        self.__testSet = testSet
        self.train(self.__trainSet,alpha,step)
        self.getPrediction()

    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))

    def train(self,trainSet,alpha,step):
        # random assign values to weights
        self.__weight = np.random.rand(len(self.__attributes),1)

        target = np.array([trainSet[i][-1] for i in range(len(trainSet))])
        X = np.array([trainSet[i][:-1] for i in range(len(trainSet))])

        for i in range(step):
            f_out = self.sigmoid(X.dot(self.__weight))
            print(f_out)
          #  print(target)
            error = [target[j]-f_out[j] for j in range(len(target))]
            # bug
            self.__weight = self.__weight + alpha*np.transpose(X).dot(error)

    def predict(self,array):
        out =  array.dot(self.__weight)
        if(out>0.5):
            return 1
        else:
            return 0



    def getAccuracy(self, testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        # print(correct)
        return (correct / float(len(testSet))) * 100.0

    def getPrediction(self):
        predictions = []
        for i in range(len(self.__testSet)):
            result = self.predict(self.__testSet[i][:-1])
            predictions.append(result)

        # print(result)
        accuracy = self.getAccuracy(self.__testSet, predictions)
        print(accuracy)






attri,trainset = tk.readDataSet("../dataset/transfusion.csv")
print(attri)
print(trainset)

#logi_test = LogisticRegression(attri,trainset,trainset)
