## 工作日志

### 时间线
1. April.23 - April.29: Naive Bayes and Knn

### dataset记录
dataset 建议每次附加一个 data description文件
1. iris.txt
2. car.txt
3. playtennis
4. sunburnt



### 流水记录
#### April 28th
1. 每个算法写成一个类,那么每次初始化的时候传入文件名,
    -  __init__函数需要执行哪些步骤? 需要执行train么? validation 是否要分开?
    -  关于类必须实现的一些函数(接口?)


2. 代码容易出现的错误
1. python里不需要初始化,如果有强迫症非要初始化,在 __init()__初始化, 下面这段代码定义变量的方式是非常容易出错的
```python
class A(object):
    __a = []

    def __init__(self):

        pass

    def printA(self):
        print(self.__a)

    def append(self,c):
        self.__a.append(c)


a = A()
a.append(2)
b = A()
b.append(3)
a.printA()
```


