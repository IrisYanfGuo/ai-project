from matplotlib import pyplot as plt
import numpy as np

dict={}
dict[1]=3
dict[3]=4
dict[2]=3

list2d =[]
for i in range(10):
    t= [j for j in range(10)]
    list2d.append(t)

print(np.mean([list2d[i][1] for i in range(10)]))