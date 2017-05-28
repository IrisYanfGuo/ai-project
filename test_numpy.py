import numpy as np
from io import BytesIO

a = np.array([[2,3,1,0],[0,0,2,2],[1+1j,3,2,1]])
#print(a[0][0])
#print(np.zeros((2,3)))


b = np.arange(1,10)
c = np.linspace(0,1,10)
#print(b)
#print(c)



f = open("./dataset/iris.txt",'rb')
data= f.read()
c = np.genfromtxt(BytesIO(data),delimiter=',',comments='@')
#print(c[:,:-1])
#print(c[:,-1])

v = [1,2,3,4]
print(np.linalg.norm(v))


