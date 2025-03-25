import numpy as np

a = np.array([[1,2,3],[3,4,5]])

x = np.array([[[1, 2, 3],[3, 4, 5]],[[6, 7, 8],[9, 10, 11]]])
print(x[1,1,[1]])
print(x[0,1,[0]])

z = np.zeros(5)
print(z)
np.shape(z)
#Z2
z2 = np.zeros((4,5))
print(z2)
np.shape(z2)
#Y
Y = np.ones((2,3))
print(Y)
#F
F = np.full((7,8),11)
print(F)

x = np.linspace(0,5,10)
print(x)

x2 = np.arange(0,5,0.2)
print(x2)

a = 1
b = 6
amount = 50

nopat = np.random.randint(a,b+1,amount)
print(nopat)

x = np.random.randn(100)
print(x)