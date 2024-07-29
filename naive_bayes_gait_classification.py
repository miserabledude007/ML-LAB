import pandas as pd
import urllib.request
import ssl
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset directly from the URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
response = urllib.request.urlopen(url)
data = response.read().decode('utf-8')

# Load dataset into DataFrame
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
           'hours_per_week', 'native_country', 'income']
df = pd.read_csv(io.StringIO(data), header=None, names=columns, na_values=' ?', skipinitialspace=True).dropna()
df = pd.get_dummies(df)

# Split data
X = df.drop(['income_>50K', 'income_<=50K'], axis=1)
y = df['income_>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train and evaluate model
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f'Accuracy score: {accuracy:.4f}')
print('\nConfusion matrix:\n', cm)
print(f'TP = {cm[1, 1]}, TN = {cm[0, 0]}, FP = {cm[0, 1]}, FN = {cm[1, 0]}')

# Plot confusion matrix
sns.heatmap(pd.DataFrame(cm, columns=['Actual Positive: 1', 'Actual Negative: 0'],
                         index=['Predict Positive: 1', 'Predict Negative: 0']),
            annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Calculate and print classification accuracy manually
TP, TN, FP, FN = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print(f'Classification accuracy : {classification_accuracy:.4f}')
