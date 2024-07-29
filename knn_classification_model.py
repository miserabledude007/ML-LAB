import ssl
import urllib.request as req
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Bypass SSL verification (Not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load and preprocess data
california_housing = fetch_california_housing()
X = california_housing.data
y = (california_housing.target > np.median(california_housing.target)).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Print confusion matrix and classification report
print("Confusion matrix\n", confusion_matrix(y_test, y_pred))
report = classification_report(y_test, y_pred, target_names=['Low', 'High'], output_dict=True)
print("\nClassification report:")
print("precision recall f1-score support")
for label, metrics in report.items():
    if label in ['Low', 'High']:
        print(f"{label} {metrics['precision']:.2f} {metrics['recall']:.2f} {metrics['f1-score']:.2f} {metrics['support']}")
print(f"avg / total {report['macro avg']['precision']:.2f} {report['macro avg']['recall']:.2f} {report['macro avg']['f1-score']:.2f} {int(report['macro avg']['support'])}")
