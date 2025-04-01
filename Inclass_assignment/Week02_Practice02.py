import numpy as np
from matplotlib import pyplot as plt

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

X = np.hstack((a, b))  # Horizontal stacking
Y = np.vstack((a, b))  # Vertical stacking

print("X:", X)
print("Y:\n", Y)

# Traverse through all elements of Y using i, j
for i in range(Y.shape[0]):  # Loop through rows
    for j in range(Y.shape[1]):  # Loop through columns
        print(f"Element ({i}, {j}): {Y[i, j]}")


x = np.array([2020, 2021, 2022, 2023, 2024, 2025])
y = np.array([100, 200, 400, 600, 800, 1000])
plt.bar(x, y, color='red')
plt.title("Sales in Millions")
plt.xlabel("Years")
plt.ylabel("Sales")
plt.show()

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
a = a.reshape(4, 3)
n,m = np.shape(a)
print(a)
for i in range(n):
    for j in range(m):
        print("Element", i, j, "is", a[i,j])

a_del = np.delete(a, 1, axis=1)
print(a_del)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
for i in np.nditer(a):
    print(i)