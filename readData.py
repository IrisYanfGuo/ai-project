#!/usr/local/bin/python3
# File Name: readData.py
# Author: Jin LI
# mail: jin.li@vub.ac.be; homtingli@gmail.com
# Created Time: 2017-03-23 09:26:23

class readData(object):
    """docstring for readData"""

    #use attributes to store attributes
    #use instances to store instances
    __attributes = []
    __instances = []
    __filename = ""

    def __init__(self, filename):
        print("start reading dataSet")
        self.__filename = filename

    #read data set from dataset file

    def readDataSet(self):
        

        #open dataset file to read data set
        try:
            f = open(self.__filename)
            line = f.readline()
            while line:
                line = line.strip()
                if line.startswith("@attr") or line.startswith("@ATTR"):
                    self.__attributes.append(line.split()[1])
                if not line.startswith("@") and len(line)>2:
                    self.__instances.append(line.strip().split(","))
                line = f.readline()
            f.close()

        except Exception as e:
            print("open file error")

        return self.__attributes,self.__instances

    def print_attri(self):
        for i in self.__attributes:
            print(i)

    def print_ins(self):
        for i in self.__instances:
            print(i)


#attr,ins = readDataSet("qqqqq.txt")
#attr,ins = readDataSet("data.txt")
#attr,ins = readDataSet("sun.txt")
#print(attr)
#print(repr(ins))

#r = readData("sun.txt")
#attr,ins = r.readDataSet()
#print(attr)

