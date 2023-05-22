from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import DatasetCreator
import csv
import os
import pandas as pd


def standardize_and_save(df, filename):
    path = "GeneratedData"
    path = os.path.join(path, filename)
    # Initialize a StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it
    standardized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Create a list to store mean and std dev values
    scaler_values = [['feature', 'mean', 'std_dev']]

    # Iterate over features
    for i, feature in enumerate(df.columns):
        scaler_values.append([feature, scaler.mean_[i], scaler.scale_[i]])

    # Write the mean and std dev values to a CSV file
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(scaler_values)

    # Return the standardized DataFrame
    return standardized_df



df = DatasetCreator.load_data('GeneratedData\\DistanceFeatures')
# Filter the DataFrame
static_df = df.loc[df['Temporality'] != 'Dynamic']

gestures = static_df.index.get_level_values('Gesture').unique()

num_classes = len(gestures)

# Create a mapping from gesture names to integers
gesture_mapping = {gesture: i for i, gesture in enumerate(gestures)}
# Create a new column in the DataFrame for the integer labels
static_df['Gesture_Label'] = static_df.index.get_level_values('Gesture').map(gesture_mapping)

# Get the feature columns
feature_columns = [col for col in static_df.columns if col.startswith('Dist')]

# Convert the features and labels to numpy arrays
inputs = static_df[feature_columns]
labels = static_df['Gesture_Label'].values

# Scale the inputs
standardized_inputs = standardize_and_save(inputs, 'scaler_values.csv')

# Initialize a Normalizer
normalizer = Normalizer()

# Fit the normalizer to the data and transform it
normalized_inputs = pd.DataFrame(normalizer.fit_transform(standardized_inputs), columns=standardized_inputs.columns)

# Split the data into a training+validation set (80%) and a test set (20%)
inputs_train_val, inputs_test, labels_train_val, labels_test = train_test_split(normalized_inputs, labels, test_size=0.2, random_state=41)

# Further split the training+validation set into a training set (75%) and a validation set (25%)
# This results in 60% training, 20% validation, and 20% testing
inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs_train_val, labels_train_val, test_size=0.25, random_state=41)

# Define the model
autoencoder = Sequential([
    Dense(64, input_shape=(15,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(15, activation='sigmoid')
])

# Compile the model
optimizer = adam_v2.Adam(lr=0.0001)
autoencoder.compile(optimizer=optimizer, loss='mse')

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, mode='min', verbose=1)

# Train the model on the training set, using the validation set for early stopping
autoencoder.fit(inputs_train, inputs_train, validation_data=(inputs_val, inputs_val), epochs=200, callbacks=[early_stopping])

# Save the model as a SavedModel
autoencoder.save('anomaly_detector', save_format='tf')

# Convert the SavedModel to ONNX using the tf2onnx command-line tool
# You'll need to have tf2onnx installed (`pip install tf2onnx`)
os.system('python -m tf2onnx.convert --saved-model anomaly_detector --output anomaly_detector.onnx')

# Evaluate the model on the test set
loss = autoencoder.evaluate(inputs_test, inputs_test)
print('Test loss:', loss)