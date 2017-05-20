import time
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
# toolkit for project including File I/O, graphs

def readDataSet(filename):
    # open dataSet file to read data set
    attributes =[]
    instances = []
    # print("start reading dataSet")
    try:
        f = open(filename)
        line = f.readline()
        while line:
            line = line.strip()
            if line.startswith("@attr") or line.startswith("@ATTR"):
                attributes.append(line.split()[1])
            if not line.startswith("@") and len(line) > 2:
                instances.append(line.strip().split(","))
            line = f.readline()
        f.close()

    except Exception as e:
        print("open file error")
    # print("reading completed!")
    return attributes, instances

#divide dataset into training and testing dataset
def splitDataSet(dataSet,split=0.66):
    trainSet = []
    testSet = []

    for i in range(len(dataSet)):
        if np.random.rand() <= split:
            trainSet.append(dataSet[i])
        else:
            testSet.append(dataSet[i])
    return trainSet,testSet

def readCsv(filename):
    # open dataSet file to read data set
    attributes =[]
    instances = []
    # print("start reading dataSet")
    try:
        f = open(filename)
        line = f.readline()
        while line:
            if line.startswith("@attr") or line.startswith("@ATTR"):
                attributes.append(line.split()[1])
            if not line.startswith("@") and len(line) > 2:
                instances.append(line.strip().split(","))
            line = f.readline()
        f.close()

    except Exception as e:
        print("open file error")
    # print("reading completed!")

    return attributes, instances

def np_read(filename,comment='#',delimit=','):
    f= open(filename,'rb')
    data = f.read()
    array= np.genfromtxt(BytesIO(data), delimiter=delimit, comments=comment)

    attribute = array[:,:-1]
    target = array[:,-1]
    return attribute,target


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


# add some method for normalisation  1. Min-max 2. z-score

def min_max(alist,min1=0,max1=1):
    min0 = min(alist)
    max0 = max(alist)

    for i in range(len(alist)):
        alist[i] = min1+(alist[i]-min0)/(max0-min0)*(max1-min1)
    return alist

def z_score(alist):
    x_mean= np.mean(alist)
    x_sigma = np.square(np.var(alist))

    for i in range(len(alist)):
        alist[i] = (alist[i]-x_mean)/x_sigma
    return alist

def normalize4mat(mat):
    '''
    normalize for 2-d matrix
    :param mat: matrix
    :return: normalised matrix
    '''

    for col in range(len(mat)):
        t = []
        for row in range(len(mat[0])):
            t.append(mat[row][col])

        t = min_max(t)
        for row in range(len(mat)):
            mat[row][col] = t[row]
    return mat

# calculate the distance for 2 lists
def dist4list(list1,list2):
    if len(list1) != len(list2):
        print("the lengths of the 2 lists are different!")
    else:
        dist = 0
        for i in range(len(list1)):
            dist += np.sqrt((list1[i]-list2[i])**2)
        return dist







