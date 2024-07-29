import ssl
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# Bypass SSL verification (Not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load and preprocess the dataset
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df["median_house_value"] /= 1000.0
test_df["median_house_value"] /= 1000.0

# Build, compile, and train the model
def build_and_train_model(learning_rate, epochs, batch_size, validation_split, feature, label):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1,)),  # Define input shape here
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    history = model.fit(train_df[feature], train_df[label], batch_size=batch_size,
                        epochs=epochs, validation_split=validation_split)
    
    # Plotting
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(history.epoch, history.history["root_mean_squared_error"], label="Train")
    plt.plot(history.epoch, history.history["val_root_mean_squared_error"], label="Validation")
    plt.title("Loss")
    plt.legend()
    plt.ylim([
        min(history.history["root_mean_squared_error"] + history.history["val_root_mean_squared_error"]) - 0.05,
        max(history.history["root_mean_squared_error"] + history.history["val_root_mean_squared_error"]) + 0.05
    ])
    plt.show()

    # Evaluate the model
    x_test = test_df[feature]
    y_test = test_df[label]
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f"Test set evaluation results:\nLoss: {results[0]:.2f}\nRoot Mean Squared Error: {results[1]:.2f}")

# Hyperparameters and features
learning_rate = 0.08
epochs = 70
batch_size = 100
validation_split = 0.2
feature = "median_income"
label = "median_house_value"

# Shuffle the training data and build & train the model
train_df = train_df.sample(frac=1, random_state=42)
build_and_train_model(learning_rate, epochs, batch_size, validation_split, feature, label)

