import time
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class generateImage(object):
    __data = []

    def __init__(self,data):
        self.__data = data
        self.__process()

    def __process(self):
        d = self.__data
        plt.bar(range(len(d)),d) #print bar

        for t in range(len(d)):
            ss = str(d[t])
            ss = ss[:4]
            plt.text(t,d[t],ss,ha='center',va='bottom')

        plt.tight_layout()   #disjust y label
        plt.xticks(range(len(d)),range(1,len(d)+1)) #set x bar label

    def draw(self):
        plt.show()

    def save(self):
        name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        name = name+".png"
        plt.savefig(name)

    def mean(self):
        return sum(self.__data)/len(self.__data)

    def std(self):
        return "std"
