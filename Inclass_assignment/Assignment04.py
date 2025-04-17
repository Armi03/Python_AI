import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Exercise 1: Regression to the mean

x_value = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in x_value:
    d1 = np.random.randint(1, 7, size=n)
    d2 = np.random.randint(1, 7, size=n)
    s = d1 + d2
    h, h2 = np.histogram(s, bins=range(2, 14))

    # Plot the histogram
    plt.bar(h2[:-1], h / n)
    plt.title(f"Sum of Two Dice Thrown {n} Times")
    plt.xlabel("Sum")
    plt.ylabel("Relative Frequency")
    plt.xticks(range(2, 13))
    plt.show()

## Exercise 2: Regression Model

df = pd.read_csv("weight-height.csv")

X = df[['Height']]
y = df['Weight']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
plt.scatter(X, y, alpha=0.5, label="Actual data")
plt.plot(X, y_pred, color='red', label="Regression line")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Linear Regression: Height vs Weight")
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")