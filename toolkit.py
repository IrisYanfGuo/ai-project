import time
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import pandas as pd

# toolkit for project including File I/O, graphs

# the return value is matrix
def readDataSet(filename):
    # open dataSet file to read data set
    attributes =[]
    instances = []
    # print("start reading dataSet")
    try:
        attributes,instances = readCsv(filename)
        instances = instances.as_matrix()
        attributes = attributes[:-1]

    except Exception as e:
        print("open file error")
    # print("reading completed!")
    return attributes, instances

# return value is data.frame
def readCsv(filename):
    df_attritutes = []
    df_instances = []

    # the attritutes include the name of the last column in the csv file
    csv = pd.read_csv(filename)

    df_attritutes = csv.columns
    for i in df_attritutes:
        if(csv[i].dtype=='object'):
            csv[i] = csv[i].astype("category")
    df_instances = csv
    return df_attritutes, df_instances



#divide dataset(matrix) into training and testing dataset(matrix)
def splitDataSet(matrix,split=0.66):
    dataSet = matrix.tolist()
    trainSet = []
    testSet = []

    for i in range(len(dataSet)):
        if np.random.rand() <= split:
            trainSet.append(dataSet[i])
        else:
            testSet.append(dataSet[i])
    return np.matrix(trainSet),np.matrix(testSet)


def draw_bar(accuracy_list,save = False):

    d= accuracy_list
    plt.bar(range(len(d)), d)

    for t in range(len(d)):
        ss = str(d[t])
        ss = ss[:4]
        plt.text(t, d[t], ss, ha='center', va='bottom')

    plt.tight_layout()  # disjust y label
    plt.xticks(range(len(d)), range(1, len(d) + 1))  # set x bar label

    if save:
        save_img()
    plt.show()


def save_img():
    name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    name = name + ".png"
    plt.savefig(name)


# calculate the distance for 2 lists
def dist4list(list1,list2):
    if len(list1) != len(list2):
        print("the lengths of the 2 lists are different!")
    else:
        dist = 0
        for i in range(len(list1)):
            dist += np.sqrt((list1[i]-list2[i])**2)
        return dist

###########################################################

###########################################################        
# LaTex：{x}_{normalization}=\frac{x-Min}{Max-Min}   min~max Normalization
def zero_one_Normalization(matrix,col_number):
    nrow,ncol = matrix.shape
    matrix = matrix.tolist()
    #print(matrix)

    for i in range(col_number):
        max_value = max(matrix[x][i] for x in range(nrow))
        min_value = min(matrix[x][i] for x in range(nrow))

        if (max_value - min_value != 0.0):
            for j in range(nrow):
                matrix[j][i] = (matrix[j][i] - min_value) / (max_value - min_value)
                #matrix[j,i] = np.true_divide((matrix[j,i] - min_value),(max_value - min_value))
    return np.matrix(matrix)

# LaTex：{x}_{normalization}=\frac{x-\mu }{\sigma }   sigma 
def Z_ScoreNormalization(matrix,col_number):
    nrow,ncol = matrix.shape
    matrix = matrix.tolist()

    for i in range(col_number): #COL
        average = np.average([matrix[x][i] for x in range(nrow)])
        sigma = np.std([matrix[x][i] for x in range(nrow)])

        if sigma != 0:
            for j in range(nrow): #row
                matrix[j][i] = (matrix[j][i] - average) / sigma

    return np.matrix(matrix)

def log_Normalization(matrix,col_number):
    nrow,ncol = matrix.shape
    for i in range(col_number): #COL
        for j in range(nrow): #row
            matrix[j,i] = np.log10(matrix[j,i])
    return matrix


def atan_Normalization(matrix,col_number):
    nrow,ncol = matrix.shape
    for i in range(col_number): #COL
        for j in range(nrow): #row
            matrix[j,i] = np.arctan(matrix[j,i]) * 2 / np.pi
    return matrix

###########################################################
#
#     start plot id3 
# Author: find the code online, 
# thanks her/him very much
#
###########################################################
#定义文本框和箭头格式
__decisionNode = dict(boxstyle="round4", fc="0.6") #定义判断节点形态
__leafNode = dict(boxstyle="square", fc="0.8") #定义叶节点形态
__arrow_args = dict(arrowstyle="<-") #定义箭头

#绘制带箭头的注解
#nodeTxt：节点的文字标注, centerPt：节点中心位置,
#parentPt：箭头起点位置（上一节点位置）, nodeType：节点属性
def __plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=__arrow_args )

#计算叶节点数
def __getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#是否是字典
            numLeafs += __getNumLeafs(secondDict[key]) #递归调用getNumLeafs
        else:   numLeafs +=1 #如果是叶节点，则叶节点+1
    return numLeafs

#计算数的层数
def __getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#是否是字典
            thisDepth = 1 + __getTreeDepth(secondDict[key]) #如果是字典，则层数加1，再递归调用getTreeDepth
        else:   thisDepth = 1
        #得到最大层数
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth



#在父子节点间填充文本信息
#cntrPt:子节点位置, parentPt：父节点位置, txtString：标注内容
def __plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

#绘制树形图
#myTree：树的字典, parentPt:父节点, nodeTxt：节点的文字标注
def __plotTree(myTree, parentPt, nodeTxt):
    numLeafs = __getNumLeafs(myTree)  #树叶节点数
    depth = __getTreeDepth(myTree)    #树的层数
    firstStr = list(myTree.keys())[0]     #节点标签
    #计算当前节点的位置
    cntrPt = (__plotTree.xOff + (1.0 + float(numLeafs))/2.0/__plotTree.totalW, __plotTree.yOff)
    __plotMidText(cntrPt, parentPt, nodeTxt) #在父子节点间填充文本信息
    __plotNode(firstStr, cntrPt, parentPt, __decisionNode) #绘制带箭头的注解
    secondDict = myTree[firstStr]
    __plotTree.yOff = __plotTree.yOff - 1.0/__plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#判断是不是字典，
            __plotTree(secondDict[key],cntrPt,str(key))        #递归绘制树形图
        else:   #如果是叶节点
            __plotTree.xOff = __plotTree.xOff + 1.0/__plotTree.totalW
            __plotNode(secondDict[key], (__plotTree.xOff, __plotTree.yOff), cntrPt, __leafNode)
            __plotMidText((__plotTree.xOff, __plotTree.yOff), cntrPt, str(key))
    __plotTree.yOff = __plotTree.yOff + 1.0/__plotTree.totalD

#创建绘图区
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    __plotTree.totalW = float(__getNumLeafs(inTree)) #树的宽度
    __plotTree.totalD = float(__getTreeDepth(inTree)) #树的深度
    __plotTree.xOff = -0.5/__plotTree.totalW; __plotTree.yOff = 1.0;
    __plotTree(inTree, (0.5,1.0), '')
    plt.show()


###########################################################