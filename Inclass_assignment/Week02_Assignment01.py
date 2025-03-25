import numpy as np
from numpy.ma.core import reshape

a = np.array([[[1,2],[3,4]]])
b = np.array([[[5,6],[7,8]]])

#substraction
Substract = a + b
print(Substract)

#deduction
Decuction = a - b
print(Decuction)

#dividing
Divide = a / b
print(Divide)

#multiplication
Multiply = a * b
print(Multiply)

#Graterthan
GreaterThan = a >= b
print(GreaterThan)

#Lessthan
LessThan = a <= 1
print(LessThan)

#added
Added = a + 1
print(Added)

#deduct
Deduct = a - 1
print(Deduct)

#Sin
Sin = (np.sin(a))
print(Sin)

#Cos
Cos = np.cos(a)
print(Cos)

#Tan
Tan = np.tan(a)
print(Tan)

#sort
Sort = np.sort(a)
print(Sort)

n = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
n1 = np.reshape(n,(3,4))
rows,cols = n1.shape
print(rows,cols)


Row1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
Row2 = Row1.reshape(4,3)
print(Row2)
print(Row2[:,0])