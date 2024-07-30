import ssl
import urllib.request as req
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Bypass SSL verification (Not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load California Housing Dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Convert regression target to binary classification target
median_value = y.median()
y_class = (y > median_value).astype(int)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred = knn.predict(X_val_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')
