import toolkit as tk
import pandas as pd
import numpy as np


# Leave one out procedure

# leave one out from dataset

def leave_1_ins(dataset, index):
    '''
    pop the dataset[index] out
    :param dataset: matrix
    :return: (dataset, one sample)
    '''
    ins = dataset[index, :]
    remain_dataset = np.delete(dataset, index, 0)
    return ins, remain_dataset


def cross_vali_dataset(dataset, cv=10):
    '''

    :param dataset: type matrix
    :param cv: the number of validation fold
    :return: K-1 length list contain the dataset
    '''
    np.random.shuffle(dataset)
    # 向下取整
    size = int(np.floor(len(dataset) / cv))
    result = []
    begin = 0
    end = size
    for i in range(cv - 1):
        testset = dataset[begin:end, :]
        trainset = np.row_stack((dataset[0:begin, :], dataset[end:, :]))
        begin = end
        end = end + size
        result.append([trainset, testset])
    result.append([ dataset[:begin, :],dataset[begin:, :]])
    return result


att, x = tk.readDataSet("./iris.csv")

a = cross_vali_dataset(x, 16)

# test part
#for i in a:
 #   print(len(i[0]),len(i[1]))
 #   print()
    # print()
    # print(i)
