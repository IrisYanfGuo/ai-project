#!/usr/local/bin/python3
# File Name: knn.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-04-21 21:20:47
import math
import operator
import random
from readData import readData

attributes = []
instances = []

#read data set from file
def readDataSet():
    global attributes,instances

    #f = open("data.txt")
    #f = open("qqqqq.arff")
    #f = open('car.txt')
    f = open('iris.txt')
    line = f.readline()
    while line:
        line = line.strip()
        if line.startswith("@attr") or line.startswith("@ATTR"):
            attributes.append(line.split()[1])
        if not line.startswith("@") and len(line)>1:
            instances.append(line.strip().split(","))
        line = f.readline()
    f.close()
    print(attributes)


#readDataSet()

def vectorDistance(data1, data2,length):
	distance = 0
	for x in range(length):
		distance += pow((data1[x] - data2[x]),2)
	return math.sqrt(distance)


#data1 = [2,2,2,'a']
#data2 = [4,4,4,'b']
#distance = vectorDistance(data1,data2,3)
#print('Distance',end=" ")
#print(distance)

def getNeighbors(trainSet, testInstance, k):
	distances = []
	length = len(testInstance) -1
	for x in range(len(trainSet)):
		dist = vectorDistance(testInstance,trainSet[x],length)
		distances.append((trainSet[x],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

#trainSet = [[2,2,2,'a'],[4,4,4,'b'],[5,4,4,'c']]
#testInstance = [5,5,5]
#k = 1
#neighbors = getNeighbors(trainSet,testInstance,k)
#print(neighbors)

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
	return sortedVotes[0][0]

#neighbors = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'c']]
#response = getResponse(neighbors)
#print(response)

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	print(correct)
	return (correct/float(len(testSet))) *100.0

#testSet = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
#predictions = ['a','a','a']
#accuracy = getAccuracy(testSet,predictions)
#print(accuracy)

def main():
	trainSet = []
	testSet = []
	split = 0.66
	readDataSet()

	for x in range(len(instances)):
		for y in range(len(attributes)):
			#print(instances[x][y])
			#print(len(attributes))
			instances[x][y] = float(instances[x][y])
		if random.random() < split:
			trainSet.append(instances[x])
		else:
			testSet.append(instances[x])

	predictions = []
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainSet,testSet[x],k)
		result = getResponse(neighbors)
		predictions.append(result)
		#print(result)

	accuracy = getAccuracy(testSet,predictions)
	print((accuracy))

main()







