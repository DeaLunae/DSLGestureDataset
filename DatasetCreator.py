import csv
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def load_data(folderName):
    data_path = folderName
    # Initialize an empty list to store the dataframes
    df_list = []

    # Loop through each participant folder
    for participant_folder in os.listdir(data_path):
        # Set the path to the participant folder
        participant_path = os.path.join(data_path, participant_folder)
        # Loop through each CSV file in the participant folder
        for csv_file in os.listdir(participant_path):
            # Set the path to the CSV file
            csv_path = os.path.join(participant_path, csv_file)
            # Load the CSV file into a dataframe
            df = pd.read_csv(csv_path, index_col=0)
            if 'column_name' in df.columns:
                df['Gesture'] = df['Gesture'].replace('¥', 'Ø')
            # Append the dataframe to the list
            df_list.append(df)

    # Concatenate the DataFrames
    concatenated_df = pd.concat(df_list)

    # Add a row index
    concatenated_df = concatenated_df.reset_index()

    # Add a row index within each group (Gesture and Participant)
    concatenated_df['Row'] = concatenated_df.groupby(['Gesture', 'Participant']).cumcount()
    # Set the desired index order
    concatenated_df = concatenated_df.set_index(['Gesture', 'Participant', 'Row'])
    # Sort the index
    concatenated_df = concatenated_df.sort_index()

    return concatenated_df


def interpolate_to_target_rows(df, target_rows=180):
    interpolated_df = pd.DataFrame()

    for (gesture, participant), group in df.groupby(['Gesture', 'Participant']):
        # Get the original 'Row' index values
        original_rows = group.index.get_level_values('Row')
        first_row, last_row = original_rows.min(), original_rows.max()

        # Generate new 'Row' index values with the target number of rows,
        # Ensuring that the first and last ones match the original data
        new_rows = np.linspace(first_row, last_row, target_rows)

        # Create the new multi-index
        new_index = pd.MultiIndex.from_product([[gesture], [participant], new_rows])

        # Reindex and interpolate
        new_group = group.reindex(new_index).interpolate(method='linear')
        interpolated_df = interpolated_df.append(new_group)

    return interpolated_df


def create_data_with_metadata():
    data_path = 'Data\\GestureData'
    generate_path = 'GeneratedData\\WithMetaData'
    # Loop through each participant folder
    for participant_folder in os.listdir(data_path):
        # Set the path to the participant folder
        participant_path = os.path.join(data_path, participant_folder)
        generated_participant_path = os.path.join(generate_path, participant_folder)
        # Loop through each CSV file in the participant folder
        for csv_file in os.listdir(participant_path):
            # Set the path to the CSV file
            csv_path = os.path.join(participant_path, csv_file)
            string_list = csv_file.split("-")[1:]

            # Load the CSV file into a dataframe
            df = pd.read_csv(csv_path, sep=";")

            # Add the relevant data
            df = df.drop(['Email', 'Framecount', 'SessionID', 'RealTimeStamp'], axis=1)
            df['Participant'] = int(participant_folder)
            df['Gesture'] = string_list[0].replace("¥", "Ø")
            df['Temporality'] = string_list[1]
            df['Handedness'] = string_list[2]
            generated_csv_path = os.path.join(generated_participant_path, f'{string_list[0]}-{string_list[1]}-{string_list[2]}.csv')
            generated_csv_path = generated_csv_path.replace("¥", "Ø")
            if not os.path.exists(generated_participant_path):
                os.mkdir(generated_participant_path)
            df.to_csv(generated_csv_path)


def create_validation_data_with_metadata():
    data_path = 'Data\\ValidationData'
    generate_path = 'GeneratedData\\ValidationWithMetaData'
    # Loop through each participant folder
    for participant_folder in os.listdir(data_path):
        # Set the path to the participant folder
        participant_path = os.path.join(data_path, participant_folder)
        generated_participant_path = os.path.join(generate_path, participant_folder)
        # Loop through each CSV file in the participant folder
        for csv_file in os.listdir(participant_path):
            # Set the path to the CSV file
            csv_path = os.path.join(participant_path, csv_file)
            string_list = csv_file.split("-")[1:]

            # Load the CSV file into a dataframe
            df = pd.read_csv(csv_path, sep=";")

            # Add the relevant data
            df = df.drop(['Email', 'Framecount', 'FrameCount', 'SessionID', 'Timestamp'], axis=1)
            df['Participant'] = int(participant_folder)
            df['Gesture'] = string_list[0].replace("¥", "Ø").replace(".csv", "")
            generated_csv_path = os.path.join(generated_participant_path, f'{string_list[0]}')
            generated_csv_path = generated_csv_path.replace("¥", "Ø")
            if not os.path.exists(generated_participant_path):
                os.mkdir(generated_participant_path)
            df.to_csv(generated_csv_path)

def save_hierachical_df(df, path):
    data_path = os.path.join('GeneratedData', path)
    local_df = df.copy()
    # make sure the DataFrame is sorted by Participant and Gesture
    local_df = local_df.sort_values(['Participant', 'Gesture'])

    # iterate over each participant
    for participant, participant_df in local_df.groupby('Participant'):
        # create a directory for this participant
        os.makedirs(os.path.join(data_path, str(participant)), exist_ok=True)

        # iterate over each gesture for this participant
        for gesture, gesture_df in participant_df.groupby('Gesture'):
            # save this gesture's data to a CSV file in the participant's directory
            gesture_df.to_csv(os.path.join(os.path.join(data_path, str(participant)), f'{gesture}-{gesture_df["Temporality"][0]}-{gesture_df["Handedness"][0]}.csv'))

def create_joint_names(local_df):
    # Filter column names that contain the dot character
    filtered_columns = [col for col in local_df.columns if '.' in col and 'Head' not in col]
    # Split the filtered column names and extract the first index
    split_column_names = list(set(col.split('.')[0] for col in filtered_columns))
    split_column_names.sort()
    return split_column_names

def create_joint_names_with_head(local_df):
    # Filter column names that contain the dot character
    filtered_columns = [col for col in local_df.columns if '.' in col in col]
    # Split the filtered column names and extract the first index
    split_column_names = list(set(col.split('.')[0] for col in filtered_columns))
    split_column_names.sort()
    return split_column_names


# Function to extract quaternion rotation and position from a row
def get_quaternion_and_position(row, joint):
    position = np.array([row[f'{joint}.pos.x'], row[f'{joint}.pos.y'], row[f'{joint}.pos.z']]).T
    rotation = np.array([row[f'{joint}.rot.x'], row[f'{joint}.rot.y'], row[f'{joint}.rot.z'], row[f'{joint}.rot.w']]).T
    return position, rotation


def local_space_process_group(args):
    group, joint_names = args

    # Get the "Head" position and rotation for the current GestureName and ParticipantID
    head_pos, head_rot = get_quaternion_and_position(group, 'Head')
    head_rotation_inverse = R.from_quat(head_rot).inv()

    # Iterate over all joints and convert worldspace to localspace
    for joint in joint_names:
        # Get joint position and rotation
        joint_pos, joint_rot = get_quaternion_and_position(group, joint)

        # Find relative position
        relative_pos = joint_pos - head_pos

        # Rotate the relative position by the inverse of the head's rotation
        rotated_relative_pos = head_rotation_inverse.apply(relative_pos)

        group[f'{joint}.pos.x'] = rotated_relative_pos[:, 0]
        group[f'{joint}.pos.y'] = rotated_relative_pos[:, 1]
        group[f'{joint}.pos.z'] = rotated_relative_pos[:, 2]

        # Find relative rotation
        joint_rotation = R.from_quat(joint_rot)
        relative_rotation = head_rotation_inverse * joint_rotation
        relative_rotation = relative_rotation.as_quat()

        group[f'{joint}.rot.x'] = relative_rotation[:, 0]
        group[f'{joint}.rot.y'] = relative_rotation[:, 1]
        group[f'{joint}.rot.z'] = relative_rotation[:, 2]
        group[f'{joint}.rot.w'] = relative_rotation[:, 3]
    return group

def convert_to_local_space(local_df):
    # Find all unique joint names in the columns
    joint_names = create_joint_names_with_head(local_df)
    result_df = local_df.copy()

    groups = list(result_df.groupby(level=[0, 1]))
    print("Generating local space conversion")
    # Use a ProcessPoolExecutor to process the groups in parallel
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(local_space_process_group, [(group, joint_names) for _, group in groups]), total=len(groups)))

    # Assign the results back to the DataFrame
    for (name, _), group in zip(groups, results):
        result_df.loc[name] = group
    return result_df

def distance_process_group(args):
    group, joint_names = args
    # Iterate over all joints and generate distances
    for joint1, joint2 in joint_names:
            joint1_position = group[[f'{joint1}.pos.x', f'{joint1}.pos.y', f'{joint1}.pos.z']].to_numpy()
            joint2_position = group[[f'{joint2}.pos.x', f'{joint2}.pos.y', f'{joint2}.pos.z']].to_numpy()
            group[f'Dist.{joint1}.{joint2}'] = np.linalg.norm(joint1_position - joint2_position, axis=1)
    return group



def generate_distance_features(local_df):
    # Find all unique joint names in the columns
    joint_pairs = [
                    # Abduction
                    ('ThumbTip', 'PinkyTip'),  # Todo: Verify that we're using Thumb to Pinky Abduction
                    ('ThumbTip', 'IndexTip'),
                    ('IndexTip', 'MiddleTip'),
                    ('MiddleTip', 'RingTip'),
                    ('RingTip', 'PinkyTip'),

                    # PIP bend
                    ('ThumbProximalJoint', 'ThumbTip'),
                    ('IndexKnuckle', 'IndexTip'),
                    ('MiddleKnuckle', 'MiddleTip'),
                    ('RingKnuckle', 'RingTip'),
                    ('PinkyKnuckle', 'PinkyTip'),

                    # Knuckle bend
                    ('Palm', 'ThumbDistalJoint'),  # Todo: Check if this is correct
                    ('Palm', 'IndexMiddleJoint'),
                    ('Palm', 'MiddleMiddleJoint'),
                    ('Palm', 'RingMiddleJoint'),
                    ('Palm', 'PinkyMiddleJoint')
    ]
    column_names = [f'Dist.{pair[0]}.{pair[1]}' for pair in joint_pairs]
    column_names.insert(0, 'Temporality')
    column_names.insert(1, 'Handedness')
    column_names.insert(2, 'FrameCount')
    column_names.insert(3, 'Timestamp')
    result_df = pd.DataFrame(index=local_df.index, columns=column_names)
    groups = list(local_df.groupby(level=[0, 1]))
    result_df['Temporality'] = local_df['Temporality'].copy()
    result_df['Handedness'] = local_df['Handedness'].copy()
    result_df['FrameCount'] = local_df['FrameCount'].copy()
    result_df['Timestamp'] = local_df['Timestamp'].copy()
    # Use a ProcessPoolExecutor to process the groups in parallel
    print('Generating distance features')
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(distance_process_group, [(group, joint_pairs) for _, group in groups]), total=len(groups)))

    # Assign the results back to the DataFrame
    for (name, _), group in zip(groups, results):
        result_df.loc[name] = group[column_names]
    print('Done')
    return result_df

def velocity_process_group(args):
    group, joint_names = args

    # Convert time to seconds and calculate the difference between each row
    time_diff = np.diff(group['Timestamp'].values / 1000)  # convert from ms to s and calculate time differences
    time_diff = np.insert(time_diff, 0, time_diff[0])  # to maintain the shape, duplicate the first element

    # Iterate over all joints and generate velocities
    for joint in joint_names:
        # Get joint position and rotation
        joint_pos, joint_rot = get_quaternion_and_position(group, joint)

        # Calculate difference for position and divide by time difference to get velocity
        diff_x = np.diff(joint_pos[:, 0]) / time_diff[1:]
        diff_y = np.diff(joint_pos[:, 1]) / time_diff[1:]
        diff_z = np.diff(joint_pos[:, 2]) / time_diff[1:]

        group.loc[group.index[1:], f'{joint}.pos.x'] = diff_x
        group.loc[group.index[1:], f'{joint}.pos.y'] = diff_y
        group.loc[group.index[1:], f'{joint}.pos.z'] = diff_z

        # Calculate difference for rotation
        joint_rot_r = R.from_quat(joint_rot[:-1])
        joint_rot_r_next = R.from_quat(joint_rot[1:])

        quaternion_diff = (joint_rot_r.inv() * joint_rot_r_next).as_quat()

        # Convert quaternion difference to Euler angles difference
        euler_diff = R.from_quat(quaternion_diff).as_euler('xyz')

        # Divide by time difference to get angular velocity
        angular_velocity = euler_diff / time_diff[:-1, None]

        group.loc[group.index[1:], f'{joint}.rot.x'] = angular_velocity[:, 0]
        group.loc[group.index[1:], f'{joint}.rot.y'] = angular_velocity[:, 1]
        group.loc[group.index[1:], f'{joint}.rot.z'] = angular_velocity[:, 2]

    return group

def convert_to_velocity(local_df):
    # Find all unique joint names in the columns
    joint_names = create_joint_names(local_df)
    result_df = local_df.copy()

    groups = list(result_df.groupby(level=[0, 1]))
    print("Generating velocities")
    # Use a ProcessPoolExecutor to process the groups in parallel
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(velocity_process_group, [(group, joint_names) for _, group in groups]), total=len(groups)))
    # Assign the results back to the DataFrame
    for (name, _), group in zip(groups, results):
        result_df.loc[name] = group
    rot_w_cols = [col for col in result_df.columns if 'rot.w' in col]
    result_df.drop(columns=rot_w_cols, inplace=True)
    result_df = result_df.reset_index()
    result_df = result_df.groupby(['Gesture', 'Participant']).apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    result_df = result_df.set_index(['Gesture', 'Participant', 'Row'])

    return result_df


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


def PerformProjectSetup():
    create_data_with_metadata()
    df = load_data('GeneratedData\\WithMetadata')
    local_space_df = convert_to_local_space(df)
    save_hierachical_df(local_space_df, 'LocalSpace')
    distance_features = generate_distance_features(df)
    save_hierachical_df(distance_features, 'DistanceFeatures')


if __name__ == '__main__':
    #PerformProjectSetup() #DO THIS WHEN YOU PULL THE PROJECT
    df = load_data('GeneratedData\\LocalSpace')
    velocity_df = convert_to_velocity(df)
    save_hierachical_df(velocity_df, 'VelocityFeatures')

    







