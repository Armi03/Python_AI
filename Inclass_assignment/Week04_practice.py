import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5])
##y = np.array([5 + 1*i for i in x])
y1 = np.array([2 * i+1 for i in x])
y2 = np.array([2 * i+2 for i in x])
y3 = np.array([2 * i+3 for i in x])

##plt.plot(x, y, marker='o', linestyle=':', color='pink', label='y = 2 + 3x')
plt.scatter(x, y1, marker='s', color='blue', label='y = 2x + 1')
plt.scatter(x, y2, marker='^', color='green', label='y = 2x + 2')
plt.scatter(x, y3, marker='D', color='orange', label='y = 2x + 3')
plt.scatter(x, y1, color='blue', marker='o', label='Observations')
plt.title("Comparison of Linear Functions")
plt.xlabel("X-Independent Variable")
plt.ylabel("Y-Dependent Variable")
plt.legend()
plt.show()



####### Regration & Calculations #######

df = pd.read_csv('linreg_data.csv',skiprows=0,names=['x','y'])

print(df.head())
print(df.columns)

xp = df['x']
yp = df['y']

x_point = np.mean(xp)
y_point = np.mean(yp)
n = len(xp)

b = (np.sum(xp * yp) - n * x_point * y_point) / (np.sum(xp ** 2) - n * x_point ** 2)
a = y_point - b * x_point

print("intercept (a):", a)
print("n is:",n)
print("slope (b):", b)


plt.scatter(xp, yp, color='pink', label='data Points')
plt.plot(xp, a + b * xp, color='black', label='regression Line')
plt.scatter(x_point, y_point, color='red', s=100, marker='o', label='(x̄, ȳ)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('regression line')
plt.show()