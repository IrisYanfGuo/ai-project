from readData import readData
# pre_process car.txt
dict_translate = {}

# deal with first and second attritube
t = 3
for i in 'vhigh,high,med,low'.strip().split(','):
    dict_translate[i] = t
    t -= 1
# deal with third attritube
dict_translate['5more'] = 5
# deal with 4 attritube
dict_translate['more'] = 5
# deal with 5th attri
t = 0
for i in 'small,med,big'.strip().split(','):
    dict_translate[i] = t
    t += 1
# deal with 6th attri
t = 0
for i in 'low,med,high'.strip().split(','):
    dict_translate[i] = t
    t += 1
# deal with target
dict_translate['unacc']=0
dict_translate['acc']=1
dict_translate['good']=2
dict_translate['vgood']=3

car = readData('../car.txt')
car_attr,car_ins = car.readDataSet()

f = open('./txtfile/car_prepro.txt','w')
f.write("@data car after preprocess\n\n")
for i in car_attr:
    f.write("@attr "+i+'\n')
f.write("@class Play\n")

f.write('\n')

for i in car_ins:
    for j in i[:-1]:
        if j in dict_translate.keys():
            f.write(str(dict_translate[j])+',')
        else:
            f.write(j+',')
    if i[-1] in dict_translate.keys():
        f.write(str(dict_translate[j]))
    else:
        f.write(j)
    f.write('\n')
f.close()

