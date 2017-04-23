from readData import readData

iris= readData("../iris.txt")

iris_attr,iris_ins = iris.readDataSet()

iris.print_attri()
