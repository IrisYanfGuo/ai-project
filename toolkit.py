import time
import matplotlib.pyplot as plt
import math
import numpy as np
# toolkit for project including File I/O, graphs

def readDataSet(filename):
    # open dataset file to read data set
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

def readCsv(filename):
    # open dataset file to read data set
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










