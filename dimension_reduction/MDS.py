import toolkit as tk
import numpy as np

class MDS(object):
    '''
    MDS: multiple Dimensional Scaling
    input: Distance matrix, _d: dimension after reduction
    '''

    def __init__(self,X):
        '''
        :param X: the m*a matrix
         1.normalisation
         2. calculate the distance and save it to dist
         3. do MDS
        '''
        self.__X= self.__normalize(X)
        self.__dist, self.__disti_,self.__dist_j,self.__dist__ = self.__dist(self.__X)
        self.__B = self.__Bmat(self.__dist,self.__disti_,self.__dist_j,self.__dist__)

        # 对B 做特征值分解 ,函数?
    def __normalize(self,X):

        for col in range(len(X[0])):
            t = []
            for row in range(len(X)):
                t.append(X[row][col])

            t = tk.min_max(t)
            for row in range(len(X)):
                self.__X[row][col] = t[row]
        return self.__X

    def __dist(self,X):
        '''
        dist_2 means pow(distance**2)
        '''
        dist_2=[]
        disti_2 = []
        dist_j2 = []
        for i in range(len(X)):
            disti_temp = 0
            for j in range(len(X)):
                dist = tk.dist4list(X[i],X[j])**2
                dist_2.append(dist)
                disti_temp+= dist
            disti_2.append(disti_temp/len(X))

        for col in range(len(dist_2)[0]):
            dist_j2 .append( sum([dist_2[i][col] for i in range(len(dist_2))])/len(X))

        return dist_2,disti_2,dist_j2,sum(dist_2)/(len(X)**2)

    def __Bmat(self,dist,disti_,dist_j,dist__):
        bmat = []
        if len(dist) != len(disti_) or len(dist)!= dist_j:
            print("invalid parameter")
        else:
            for i in range(len(dist)):
                for j in  range(len(dist[0])):
                    t = -0.5*(dist[i][j]-disti_[i]-dist_j[j]+dist__)
                    bmat.append(t)
        return bmat










