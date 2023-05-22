import os
import numpy as np
import pandas as pd

import DatasetCreator
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.callbacks import EarlyStopping


def truncate_sequences(sequences):
    # Determine the length of the shortest sequence
    min_len = min(len(seq) for seq in sequences)

    # Truncate all sequences down to the length of the shortest sequence
    truncated_sequences = []
    for seq in sequences:
        if len(seq) != min_len:
            # Calculate start and end points for truncation
            start = (len(seq) - min_len) // 2
            end = start + min_len
            truncated_sequences.append(seq[start:end])
        else:
            truncated_sequences.append(seq)
    return truncated_sequences


df = DatasetCreator.load_data('GeneratedData\\VelocityFeatures')
dist_df = DatasetCreator.load_data('GeneratedData\\DistanceFeatures')

columns_to_add = [col for col in dist_df.columns if col.startswith('Dist')]
df = pd.concat([df, dist_df[columns_to_add]], axis=1)
# Get the feature columns
gestures = df.index.get_level_values('Gesture').unique()

num_classes = len(gestures)
feature_columns = [col for col in df.columns if '.' in col and 'Head' not in col]
num_features = len(feature_columns)
dynamic_df = df[feature_columns]
# Convert the features and labels to numpy arrays

# Group the data by Gesture and Participant
groups = dynamic_df.groupby(['Gesture', 'Participant'])

# Convert each group to a numpy array and store them in a list
sequences = [group.values for _, group in groups]
sequences = truncate_sequences(sequences)
# Convert the list to a numpy array
sequences = np.array(sequences)

# Create labels for each sequence based on the 'Gesture' level of the group's indices
gesture_mapping = {gesture: i for i, gesture in enumerate(dynamic_df.index.get_level_values('Gesture').unique())}
labels = np.array([gesture_mapping[group_index[0]] for group_index, _ in groups])


# Now you have your sequences and labels. The next step is to split them into training and test sets.
# Split the data into a training+validation set (80%) and a test set (20%)
sequences_train_val, sequences_test, labels_train_val, labels_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Further split the training+validation set into a training set (75%) and a validation set (25%)
# This results in 60% training, 20% validation, and 20% testing
sequences_train, sequences_val, labels_train, labels_val = train_test_split(sequences_train_val, labels_train_val, test_size=0.25, random_state=42)

# Define the model
model = Sequential([
    GRU(64, input_shape=(90, num_features)),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=10, mode='max', verbose=1)

# Train the model on the training set, using the validation set for early stopping
model.fit(sequences_train, labels_train, validation_data=(sequences_val, labels_val), epochs=200, callbacks=[early_stopping])

# Save the model as a SavedModel
model.save('model', save_format='tf')

# Convert the SavedModel to ONNX using the tf2onnx command-line tool
# You'll need to have tf2onnx installed (`pip install tf2onnx`)
os.system('python -m tf2onnx.convert --saved-model model --output model.onnx')
# Evaluate the model on the test set
loss, accuracy = model.evaluate(sequences_test, labels_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
