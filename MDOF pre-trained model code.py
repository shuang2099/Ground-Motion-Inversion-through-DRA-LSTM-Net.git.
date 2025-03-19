import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------------
# Function to load .mat files
def load_mat_file(file_path, variable):
    data = loadmat(file_path)
    return data[variable]


# ------------------------------------------------------------------
# Load normalization parameters saved during training
mean = np.load('meanAtt.npy')
std = np.load('stdAtt.npy')

# ------------------------------------------------------------------
# Load test data (adjust file names and variable names as required)
# For example, if your variable is 'DOF_3' for inputs and 'ground_motion' for targets:
X_test = load_mat_file('X_test_1.mat', 'DOF_3')  # Change 'X_test.mat' & variable as needed
y_test = load_mat_file('y_test_1.mat', 'ground_motion')  # Change 'y_test.mat' & variable as needed

# Normalize test data using training parameters
X_test = (X_test - mean) / std

# Reshape test data to match the LSTM input shape: (samples, timesteps, features)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# ------------------------------------------------------------------
# Define the custom Attention layer (same as used in training)
from keras.layers import Layer


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


# Dictionary for custom objects needed to load the model
custom_objects = {'Attention': Attention}

# ------------------------------------------------------------------
# Load the trained model
model = load_model('trained_modelAtt.h5', custom_objects=custom_objects)

# ------------------------------------------------------------------
# Generate predictions on test data
y_test_pred = model.predict(X_test).flatten()

# ------------------------------------------------------------------
# Calculate performance metrics
corr_coeff = np.corrcoef(y_test.flatten(), y_test_pred)[0, 1]
mse = mean_squared_error(y_test.flatten(), y_test_pred)
max_abs_error = np.max(np.abs(y_test.flatten() - y_test_pred))

print("Test Correlation Coefficient:", corr_coeff)
print("Test Mean Squared Error:", mse)
print("Test Maximum Absolute Error:", max_abs_error)

# ------------------------------------------------------------------
# Plotting Actual vs. Predicted values for test data
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten(), label='Actual')
plt.plot(y_test_pred, label='Predicted')
plt.title('Test Data Predictions')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------------
# Optional: Plot positive predictions with best fit line
def plot_positive_predictions(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    # Ensure shapes are compatible
    predicted = predicted.reshape(actual.shape)
    # Mask for samples where both actual and predicted values are positive
    mask = (actual > 0) & (predicted > 0)
    actual_positive = actual[mask].flatten()
    predicted_positive = predicted[mask].flatten()
    plt.scatter(actual_positive, predicted_positive, edgecolors='k', s=20, label='Predicted')

    # Linear fit for positive values
    if actual_positive.size > 0:
        z = np.polyfit(actual_positive, predicted_positive, 1)
        p = np.poly1d(z)
        line_space = np.linspace(0, actual_positive.max(), num=500)
        plt.plot(line_space, p(line_space), "b--", linewidth=1, label='Best Fit')

    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function for test data predictions (if applicable)
plot_positive_predictions(y_test, y_test_pred, 'Test Data Predictions - Positive Values Only')
