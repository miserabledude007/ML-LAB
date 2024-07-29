import pandas as pd
import ssl
import urllib.request as req

# Bypass SSL verification (Not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Boston_Housing/Training_set_boston.csv"
data = pd.read_csv(url)

# The rest of your code
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Prepare data
X, y = data.drop('MEDV', axis=1), data.MEDV

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Evaluate model
print("Using Linear Regression")
print(f"Training data accuracy: {r2_score(y_train, model.predict(X_train))}")
print(f"Testing data accuracy: {r2_score(y_test, model.predict(X_test))}")
