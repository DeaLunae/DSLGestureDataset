import pandas as pd
import csv
import os
import DatasetCreator
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dropout


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

# Initialize a StandardScaler
standardized_inputs = DatasetCreator.standardize_and_save(inputs, 'scaler_values.csv').values


# Split the data into a training+validation set (80%) and a test set (20%)
inputs_train_val, inputs_test, labels_train_val, labels_test = train_test_split(standardized_inputs, labels, test_size=0.2, random_state=42)

# Further split the training+validation set into a training set (75%) and a validation set (25%)
# This results in 60% training, 20% validation, and 20% testing
inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs_train_val, labels_train_val, test_size=0.25, random_state=42)

# Define the model
model = Sequential([
    Dense(64, input_shape=(15,), activation='relu'),
    Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=10, mode='max', verbose=1)

# Train the model on the training set, using the validation set for early stopping
model.fit(inputs_train, labels_train, validation_data=(inputs_val, labels_val), epochs=200, callbacks=[early_stopping])

# Save the model as a SavedModel
model.save('model', save_format='tf')

# Convert the SavedModel to ONNX using the tf2onnx command-line tool
# You'll need to have tf2onnx installed (`pip install tf2onnx`)
os.system('python -m tf2onnx.convert --saved-model model --output model.onnx')
# Evaluate the model on the test set
loss, accuracy = model.evaluate(inputs_test, labels_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
