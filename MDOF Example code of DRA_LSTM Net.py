import seaborn as seaborn
import tensorflow as tf
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Layer
from scipy.io import loadmat
############################# Load data

def load_mat_file(file_path, variable):
    data = loadmat(file_path)
    return data[variable]

# Selecting a specific DOF
selected_dof = 'DOF_3'

# Load data from .mat files for the selected DOF
X_train = load_mat_file('X_train.mat', selected_dof)
X_val = load_mat_file('X_val.mat', selected_dof)
X_test = load_mat_file('X_test_71.mat', selected_dof)

# Assuming the target variable is stored with this name in the .mat files
y_train = load_mat_file('y_train.mat', 'ground_motion')
y_val = load_mat_file('y_val.mat', 'ground_motion')
y_test = load_mat_file('y_test_71.mat', 'ground_motion')

# Normalize data
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_val = (X_val - mean) / std

# Reshape data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Define Attention layer
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

# Define the step decay function for learning rate
initial_lr = 0.001
decay = 0.005
initial_epoch = 0

def step_decay(epoch):
    lr = initial_lr / (1 + decay * (initial_epoch + epoch))
    return max(lr, 5e-5)

import os
# Directory for saving the files
save_dir = '../save'
# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create LearningRateScheduler and ModelCheckpoint callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint_filepath = './save/checkpoint.ckpt'
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                             save_best_only=True, verbose=0, save_weights_only=True,
                             mode='auto', save_freq='epoch')
################################################################# Define LSTM network with Attention mechanism
model = Sequential()
number_of_layers = 7  # Adjust the number of layers as needed
# sigmoid, tanh, relu, LeakyReLU, Swish, ELU, gelu
for i in range(number_of_layers - 1):
    if i == 0:
        # For the first LSTM layer, set the input_shape
        model.add(LSTM(256, return_sequences=True, recurrent_activation='sigmoid',
                       recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.01),
                       activation='elu', input_shape=(X_train.shape[1], X_train.shape[2])))
    else:
        # For subsequent LSTM layers, input_shape is not needed
        model.add(LSTM(256, return_sequences=True, recurrent_activation='sigmoid',
                       recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.01),
                       activation='elu'))
    model.add(Attention(return_sequences=True))

# Final LSTM layer
model.add(LSTM(256, recurrent_activation='sigmoid',return_sequences=False,
               recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.01),
               activation='elu'))

# Output layer - adjust as per your requirement
model.add(Dense(30))  # Change the number of units according to your output

# Compile the model with the customized Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999,
                                                 epsilon=1e-07, amsgrad=False),
              loss='mae',
              metrics=['mae'])

################################################################### Train the model with the new callbacks
history = model.fit(X_train, y_train, epochs=500, batch_size=256, validation_data=(X_val, y_val),
                    callbacks=[lrate, checkpoint], verbose=2)


################### for 1st result loss vs epoch
import pandas as pd

# Convert training history to a pandas DataFrame
loss_data = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),  # Assuming 1-based indexing for human readability
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})

# Save the DataFrame to a CSV file
loss_data.to_csv('epoch_vs_loss.csv', index=False)


########################################################## Find the minimum validation loss and its corresponding epoch
min_vloss = min(history.history['val_loss'])
min_vloss_epoch = history.history['val_loss'].index(min_vloss)
print('Reached minimum validation loss %e at epoch %d.' % (min_vloss, min_vloss_epoch))


# Save the loss history to text files
np.savetxt(os.path.join(save_dir, 'loss1.txt'), history.history['loss'])
np.savetxt(os.path.join(save_dir, 'vloss1.txt'), history.history['val_loss'])

################################################################################# Evaluate model performance
# Load the best weights into the model
model.load_weights(checkpoint_filepath)

train_loss = model.evaluate(X_train, y_train, verbose=0)
val_loss = model.evaluate(X_val, y_val, verbose=0)  # Evaluating on the validation dataset
test_loss = model.evaluate(X_test, y_test, verbose=0)

print('Train Loss:', train_loss)
print('Validation Loss:', val_loss)  # Outputting the validation loss
print('Test Loss:', test_loss)

#####################################################################  Plots

# Make predictions
y_train_pred = model.predict(X_train).flatten()
y_val_pred = model.predict(X_val).flatten()
y_test_pred = model.predict(X_test).flatten()


###################### save prediction results for result graphs 2 and 3
# Reshape predictions to match the original data's shape, if necessary
y_train_predd = y_train_pred.reshape(y_train.shape)
y_val_predd = y_val_pred.reshape(y_val.shape)
y_test_predd = y_test_pred.reshape(y_test.shape)

preddictions_data = {
    "y_train_predd": y_train_predd,
    "y_val_predd": y_val_predd,
    "y_test_predd":y_test_predd
}
from scipy.io import savemat
savemat("preddiction_results.mat", preddictions_data)

########################################################################

# Calculate correlation coefficients
train_corr = np.corrcoef(y_train.flatten(), y_train_pred)[0, 1]
val_corr = np.corrcoef(y_val.flatten(), y_val_pred)[0, 1]
test_corr = np.corrcoef(y_test.flatten(), y_test_pred)[0, 1]

print('Train Correlation Coefficient:', train_corr)
print('Validation Correlation Coefficient:', val_corr)
print('Test Correlation Coefficient:', test_corr)


import numpy as np
from sklearn.metrics import mean_squared_error

# Assuming y_test_pred is your predictions for the test dataset
# and y_test is the actual values

# Calculate Maximum Absolute Error (MaxAE) for test data manually
max_ae = np.max(np.abs(y_test.flatten() - y_test_pred))
print('Maximum Absolute Error (MaxAE) on Test Dataset:', max_ae)

# Calculate Mean Squared Error (MSE) for test data
mse = mean_squared_error(y_test.flatten(), y_test_pred)
print('Mean Squared Error (MSE) on Test Dataset:', mse)



# Plot loss history on log scale
plt.figure(figsize=(12, 6))
plt.semilogy(history.history['loss'], label='Train Loss')
plt.semilogy(history.history['val_loss'], label='Validation Loss')
plt.title('Loss History on Log Scale')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.show()

# Plot training predictions
plt.figure(figsize=(12, 6))
plt.plot(y_train.flatten(), label='Actual')
plt.plot(y_train_pred, label='Predicted')
plt.title('Training Data Predictions')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot validation predictions
plt.figure(figsize=(12, 6))
plt.plot(y_val.flatten(), label='Actual')
plt.plot(y_val_pred, label='Predicted')
plt.title('Validation Data Predictions')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot test predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten(), label='Actual')
plt.plot(y_test_pred, label='Predicted')
plt.title('Test Data Predictions')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()




import matplotlib.pyplot as plt
import numpy as np

############### plot best fit line for results graph 2 and 3
def plot_positive_predictions(actual, predicted, title):
    plt.figure(figsize=(12, 6))

    # Ensure predicted is not flattened (if your predicted values are per-sample and not per-feature)
    predicted = predicted.reshape(actual.shape)

    # Mask for samples where all predicted values are positive
    mask = np.all((actual > 0) & (predicted > 0), axis=1)

    # Filter both actual and predicted arrays
    actual_positive = actual[mask]
    predicted_positive = predicted[mask]

    # Flatten the arrays to create a 1D array for plotting
    actual_positive_flat = actual_positive.flatten()
    predicted_positive_flat = predicted_positive.flatten()

    # Scatter plot of positive values only
    plt.scatter(actual_positive_flat, predicted_positive_flat, color='orange', edgecolors='k', s=20, label='Predicted')

    # Linear fit for positive values only
    z = np.polyfit(actual_positive_flat, predicted_positive_flat, 1)
    p = np.poly1d(z)

    # Create a line space for the positive range only
    line_space = np.linspace(0, actual_positive_flat.max(), num=500)

    # Plot the best fit line for the positive range only
    plt.plot(line_space, p(line_space), "b--", linewidth=1)

    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim(left=0)  # Set the x-axis to start at 0
    plt.ylim(bottom=0)  # Set the y-axis to start at 0
    plt.legend()
    plt.grid(True)
    plt.show()

# Then call your function with correctly shaped data
plot_positive_predictions(y_train, y_train_pred, 'Training Data Predictions - Positive Values Only')
# Call the function for validation data
plot_positive_predictions(y_val, y_val_pred, 'Validation Data Predictions - Positive Values Only')




################################################################################## saving training process
np.save('meanAtt.npy', mean)  # Saving the mean
np.save('stdAtt.npy', std)    # Saving the standard deviation
model.save('trained_modelAtt.h5')  # Saves the model in HDF5 format
import json
# Save the model architecture in JSON format
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)

import numpy as np

# Convert all float32 values in the history to Python float
def convert_float32_to_float(history):
    for key in history.keys():
        history[key] = [float(val) if isinstance(val, np.float32) else val for val in history[key]]
    return history

# Convert the training history
converted_history = convert_float32_to_float(history.history)

# Save the converted training history to a JSON file
with open("training_history.json", "w") as json_file:
    json.dump(converted_history, json_file)


# Save a file with environment details
environment_details = {
    "python_version": "3.11.1",
    "keras_version": "2.12.0",
    "tensorflow_version": "2.12.0"
}
with open("environment_details.json", "w") as json_file:
    json.dump(environment_details, json_file)

# training loss< 0.0082 and validation loss< 0.0314
# Rtrain=0.998 Rvali=0.952