import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Step 1: Read the CSV file with correct delimiter
df = pd.read_csv("bank.csv", delimiter=';')
print("Step 1 - First 5 rows of the dataset:")
print(df.head())
print("\nData types:\n", df.dtypes)

# Step 2: Select specific columns to df2
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print("step02 =", df2.head())


# Step 3: Convert 'y' to binary, then convert categoricals to dummy variables
df2['y'] = df2['y'].apply(lambda x: 1 if x == 'yes' else 0)
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

# Step 4: Heatmap of correlation
plt.figure(figsize=(15, 10))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Step 4 - Correlation Heatmap of df3")
plt.show()

"""
From the heatmap indicates:
- 'y' signifies partial association with certain occupational and housing type.
- Most of the variables are weak or have no strong correlation, as is normal with categorical attributes.
"""

# Step 5: Separate target and features
y = df3['y']
X = df3.drop(columns=['y'])

# Step 6: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Step 8: Logistic Regression Results
print("\nStep 8 - Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

# Step 9: K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Step 10: KNN Results
print("\nStep 10 - KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("KNN Accuracy (k=3):", accuracy_score(y_test, y_pred_knn))

"""
Comparison:
- Logistic regression typically performs better with high-dimensional categorical data.
- KNN may perform worse if there is a lot of sparse or dummy data (as here).
- Logistic regression accuracy is usually higher and more stable.
"""
