#!/usr/local/bin/python3
# File Name: kmeans.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-05-19 10:34:20
import toolkit
import numpy as np
import matplotlib.pyplot as plt


class kmeans(object):
    """docstring for kmeans"""
    # read trainSet, "attributes", and the number of cluster
    def __init__(self,attributes,trainSet,k=4):
        self.__predictions = []
        self.__k = k
        self.__attributes = attributes
        self.__trainSet = trainSet
        #self.__training()
        


    def __getEuclideanDistance(self,data1,data2):
        return np.sqrt(np.sum(np.power(data1 - data2, 2)))

    #print(np.random.rank(k))
    def __training(self):
        nrow, ncol = np.shape(self.__trainSet)
        cluster_points = np.mat(np.zeros((self.__k,ncol)))
        for j in range(ncol):
            min_value = np.min(self.__trainSet[:,j])
            max_value = np.max(self.__trainSet[:,j])
            # find random cluster points, 
            # add row by row
            cluster_points[:,j] = min_value + (max_value-min_value)*np.random.rand(self.__k,1)
        print(cluster_points)
        
        clusterTable = np.mat(np.zeros((nrow,3))) #record index, dist, cluster
        for i in range(nrow):
            clusterTable[i,0] = i

        flag = True

        while flag:
            flag = False
            for i in range(nrow):
                min_dist = np.inf 
                min_index = -1
                for j in range(self.__k):
                    dist = self.__getEuclideanDistance(cluster_points[j,:],self.__trainSet[i,:])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = j
                if clusterTable[i,1] != min_index:
                    flag = True
                    clusterTable[i,:] = i,min_index,np.power(min_dist,2)

            for kk in range(self.__k):
                new_cluster = np.mat(np.zeros((1,ncol)))
                count = 0
                for j in range(nrow):
                    if (clusterTable[j,1]) == kk:
                        count = count + 1
                        new_cluster = new_cluster + self.__trainSet[j,:]
                new_cluster = new_cluster / count
                for r in range(ncol):
                    cluster_points[kk,r] = new_cluster[0,r]

        return cluster_points,clusterTable

    def getPrediction(self):
        cluster_points,clusterTable = self.__training()
        return cluster_points,clusterTable
        


