import pandas as pd
import ssl
import urllib.request as req
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Bypass SSL verification (Not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load and shuffle dataset
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv").sample(frac=1, random_state=42)
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Normalize data
train_mean, train_std = train_df.mean(), train_df.std()
test_mean, test_std = test_df.mean(), test_df.std()
train_df_norm = (train_df - train_mean) / train_std
test_df_norm = (test_df - test_mean) / test_std

# Define binary labels
threshold = 265000
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)

# Define and create the model
def create_model(input_shapes, lr):
    inputs = {name: tf.keras.Input(shape=(shape,), name=name) for name, shape in input_shapes.items()}
    concatenated_inputs = layers.Concatenate()(list(inputs.values()))
    outputs = layers.Dense(1, activation='sigmoid')(concatenated_inputs)
    model = tf.keras.Model(inputs=list(inputs.values()), outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.35),
                      tf.keras.metrics.Precision(thresholds=0.35, name='precision'),
                      tf.keras.metrics.Recall(thresholds=0.35, name='recall'),
                      tf.keras.metrics.AUC(name='auc')
                  ])
    return model

# Prepare data for training and evaluation
input_shapes = {'median_income': 1, 'total_rooms': 1}
features_train = {name: np.array(train_df_norm[name]) for name in input_shapes}
features_test = {name: np.array(test_df_norm[name]) for name in input_shapes}
label_train = np.array(train_df_norm.pop('median_house_value_is_high'))
label_test = np.array(test_df_norm.pop('median_house_value_is_high'))

# Create and train the model
model = create_model(input_shapes, lr=0.001)
history = model.fit(x=features_train, y=label_train, epochs=30, batch_size=100, verbose=0)

# Evaluate model
y_pred_prob = model.predict(features_test).ravel()
fpr, tpr, _ = roc_curve(label_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Print and plot results
print(f"Final Metric Values - AUC: {roc_auc:.3f}")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc='lower right')
plt.show()
