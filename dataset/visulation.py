import toolkit as tk
import pandas as pd
from pandas.plotting import radviz,parallel_coordinates
import matplotlib.pyplot as plt

data = pd.read_csv("./wine.csv")

plt.figure()
parallel_coordinates(data,"1065")
tk.save_img()
plt.show()


#car = pd.read_csv("./car.csv")
#print(car)
#car.plot(kind='box')
#plt.show()