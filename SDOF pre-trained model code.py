import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.layers import Layer
from scipy.io import loadmat, savemat
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

#----------------------------
# Custom Attention layer definition (training ke dauran use hua tha)
class Attention(Layer):
    def __init__(self, return_sequences=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"return_sequences": self.return_sequences})
        return config

#----------------------------
# Helper function to load .mat files
def load_mat_file(file_path, variable='data'):
    data = loadmat(file_path)
    return data[variable]

#----------------------------
# File names (sabhi files current directory mein hain)
model_path = 'trained_modelAtt.h5'
mean_file = 'meanAtt.npy'
std_file = 'stdAtt.npy'
test_data_file = 'X_test_1.mat'
actual_labels_file = 'y_test_1.mat'  # Agar actual labels available nahi hain, to is line ko comment kar dein

#----------------------------
# Load the trained model with custom Attention layer
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model file not found: {model_path}")

model = load_model(model_path, custom_objects={'Attention': Attention})
print("Model loaded successfully.")

# Load normalization parameters
if not os.path.exists(mean_file) or not os.path.exists(std_file):
    raise FileNotFoundError("Normalization parameter files not found.")
mean = np.load(mean_file)
std = np.load(std_file)
print("Normalization parameters loaded.")

#----------------------------
# Load test data from .mat file and preprocess
if not os.path.exists(test_data_file):
    raise FileNotFoundError(f"Test data file not found: {test_data_file}")

X_test = load_mat_file(test_data_file, variable='data')
# Normalize test data using saved mean and std
X_test = (X_test - mean) / std
# Reshape for LSTM input: (samples, 1, features)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print("Test data loaded and preprocessed.")

#----------------------------
# Generate predictions
y_test_pred = model.predict(X_test).flatten()
print("Predictions generated.")

#----------------------------
# Load actual labels if available
actual = None
if os.path.exists(actual_labels_file):
    actual = load_mat_file(actual_labels_file, variable='data')
    actual = actual.flatten()  # Flatten the array if needed
    print("Actual labels loaded.")

#----------------------------
# Plot Actual vs Predicted (agar actual labels available hon)
plt.figure(figsize=(12, 6))
if actual is not None:
    plt.plot(actual, label='Actual', linewidth=2)
plt.plot(y_test_pred, label='Predicted', linestyle='--')
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#----------------------------
# Calculate and print error metrics if actual labels are available
if actual is not None:
    mae = mean_absolute_error(actual, y_test_pred)
    mse = mean_squared_error(actual, y_test_pred)
    corr_coef = np.corrcoef(actual, y_test_pred)[0, 1]
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Correlation Coefficient: {corr_coef}")

#----------------------------
# Function to plot positive predictions with best-fit line
def plot_positive_predictions(actual_vals, predicted_vals, title):
    plt.figure(figsize=(12, 6))
    # Ensure predicted_vals has the same shape as actual_vals
    predicted_vals = predicted_vals.reshape(actual_vals.shape)
    # Filter samples where both actual and predicted values are positive
    mask = (actual_vals > 0) & (predicted_vals > 0)
    actual_positive = actual_vals[mask]
    predicted_positive = predicted_vals[mask]
    plt.scatter(actual_positive, predicted_positive, color='orange', edgecolors='k', s=20, label='Data Points')
    if len(actual_positive) > 1:
        z = np.polyfit(actual_positive, predicted_positive, 1)
        p = np.poly1d(z)
        line_space = np.linspace(0, actual_positive.max(), num=500)
        plt.plot(line_space, p(line_space), "b--", linewidth=1, label="Best Fit Line")
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.show()

if actual is not None:
    plot_positive_predictions(actual, y_test_pred, "Positive Values: Actual vs Predicted")

#----------------------------
# Save predictions to a .mat file
savemat("test_predictions.mat", {"y_test_pred": y_test_pred})
print("Predictions saved to 'test_predictions.mat'.")

#----------------------------
# Optionally, load and display environment details if saved earlier
env_details_file = "environment_details.json"
if os.path.exists(env_details_file):
    with open(env_details_file, "r") as f:
        env_details = json.load(f)
    print("Environment Details:", env_details)
