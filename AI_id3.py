#!/usr/local/bin/python3
# File Name: id3.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-03-27 22:10:06
import numpy as np
import operator
import toolkit as tk 

#attributes[] is to store attributes
#instances[] is used to store data set line by line

class id3(object):
    """docstring for id3"""
    def __init__(self,attributes,trainSet):
        self.__attributes = attributes
        self.__trainSet = trainSet
        self.__testSet = []
        self.__predictions = []
        self.__tree = {}

    #calculate entropy
    def __Entropy(self,datas):
        data_entropy = 0.0
        freq_list = {}
        for data in datas:
            freq_list[data[-1]] = 1.0 + freq_list.get(data[-1],0.0)

        for freq in freq_list.values():
            data_entropy += -freq/len(datas) * np.log2(freq/len(datas))

        #print()
        return data_entropy

    #Divide the data set, 
    #"attrPosition" is the position of eigenvalue
    #"value" is the value that corresponding to the eigenvalue.
    #filter the sample that eigenvalue==value 
    def __selectDataSet(self,datas,attrPosition,value):
        result = []
        for data in datas:
            if data[attrPosition]==value:
                result.append(data[:attrPosition]+data[attrPosition+1:])

        return result

    # calculate the infomation Gain
    def __informationGain(self,datas):
        attrNum = len(datas[0]) - 1
        baseEntroy = self.__Entropy(datas)
        bestInformationGain = 0.0
        bestAttrPosition = -1

        for i in range(attrNum):
            attrSet = set()
            for data in datas:
                attrSet.add(data[i])
            newEntropy = 0.0
            for value in attrSet:
                subData = self.__selectDataSet(datas,i,value)
                prob=len(subData)/float(len(datas))  #Sv/S
                #Gain(S,A)=Entroy(S)-Sum((Sv/S)*Entroy(Sv))
                newEntropy += prob*self.__Entropy(subData)
            informationGain = baseEntroy - newEntropy
            if (informationGain>bestInformationGain):
                bestInformationGain = informationGain
                bestAttrPosition = i
        return bestAttrPosition,bestInformationGain
    ####
    # Few obey the majority
    def __count_most_data(self,classList):
        classCount={}#每种类别包含的个数
        for i in classList:
            classCount[i[0]] = 1 + classCount.get(i[0],0)

        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        #sortedClassCount = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
        return sortedClassCount[0][0]

    # train Data Set,
    # return a dictionay 
    def training(self):
        attrs = self.__attributes
        self.__tree = self.__training(self.__trainSet,self.__attributes[:])
        return self.__tree

    # input test Data set and get corresponding results
    def getResult(self, testSet):
        self.__predictions = []
        self.__testSet = testSet
        for i in range(len(testSet)):
            res = self.__getResult(self.__tree.copy(),testSet[i][:])
            self.__predictions.append(res)

        return self.__predictions

    def getAccuracy(self):
        # print accuracy
        if (len(self.__testSet)==0 or len(self.__predictions)==0):
            return None
        if (len(self.__testSet[0])==len(self.__trainSet[0])):
            count = 0
            for i in range(len(self.__testSet)):
                if self.__testSet[i][-1] == self.__predictions[i]:
                    count = count + 1
        return float(count)/len(self.__testSet)*100

    # use iteration
    def __getResult(self,tree,data):
        firstLabel = list(tree.keys())[0] # the name of first label
        secondDict = tree[firstLabel]   # the corresponding value
        attrIndex = self.__attributes.index(firstLabel) # attr 's index
        classLabel = None
        for key in secondDict.keys():
            if data[attrIndex]==key:
                if type(secondDict[key]).__name__=="dict":
                    classLabel = self.__getResult(secondDict[key],data)
                else:
                    classLabel=secondDict[key]
        return classLabel

    # use iteration to train
    def __training(self,datas,attrs):
        classList = []
        for data in datas:
            classList.append(data[-1])
        if len(set(classList))==1:  #if only one leaf, return
            return classList[0]
        
        ########
        #print(len(datas[0]))
        if len(datas[0])==1: #The attribute is completely exhausted
            return self.__count_most_data(datas)
        ########
        bestAttrPosition,bestInformationGain = self.__informationGain(datas)
        treeLabel = attrs[bestAttrPosition]
        #print(treeLabel,end=": ")
        #print(bestInformationGain)

        # to build tree
        myTree = {treeLabel:{}}
        del(attrs[bestAttrPosition])


        subValues = set()
        for temp in datas:
            subValues.add(temp[bestAttrPosition])
        for temp in subValues:
            subTreeLabel = attrs[:]
            myTree[treeLabel][temp] = self.__training(
                self.__selectDataSet(datas,bestAttrPosition,temp),subTreeLabel)
        return myTree
