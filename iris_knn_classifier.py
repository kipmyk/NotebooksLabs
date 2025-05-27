#Step 3: Build a Machine Learning Model

from sklearn.model_selection import train_test_split
#Step 1: Load the Data

import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                      test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Step 2: Train the model on the training data
knn.fit(X_train, y_train)

# Step 3: Use the trained model to make predictions on the test data
y_pred = knn.predict(X_test)

# Step 4: Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Step 5: Print the accuracy as a percentage
print(f'Accuracy: {accuracy * 100:.2f}%')