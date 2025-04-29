import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
'''
# Load dataset
df = pd.read_csv('Admission_Predict.csv')

# Use only selected features and target
X = df[["CGPA", "GRE Score"]]
y = df["Chance of Admit "]  # Make sure there's a space after "Admit" in the CSV

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Normalize
scaler_norm = MinMaxScaler()
X_train_norm = scaler_norm.fit_transform(X_train)
X_test_norm = scaler_norm.transform(X_test)

# Standardize
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# Use KNeighborsRegressor
model = neighbors.KNeighborsRegressor(n_neighbors=5)

# Fit and score: Original
model.fit(X_train, y_train)
print("R2 =", model.score(X_test, y_test))

# Fit and score: Normalized
model.fit(X_train_norm, y_train)
print("R2 (norm) =", model.score(X_test_norm, y_test))

# Fit and score: Standardized
model.fit(X_train_std, y_train)
print("R2 (std) =", model.score(X_test_std, y_test))
'''


df = pd.read_csv('data_banknote_authentication.csv')
print(df.head())

X = df.drop(['class'],axis=1)
y = df['class']
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))