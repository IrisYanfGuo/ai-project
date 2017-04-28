from knn import knn
from visualization import generateImage

temp = []


k = knn("iris.txt",0.7,3)
for i in range(40):
    temp.append(k.training())


g = generateImage(temp)
print(g.mean())
g.draw()

#g.save() # save 和draw 不兼容, 自带问题
