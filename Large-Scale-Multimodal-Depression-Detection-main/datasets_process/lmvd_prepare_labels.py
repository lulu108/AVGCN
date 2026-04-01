#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Function to format indices based on their value
def format_index(i):
    if i <= 601:
        return f'{i:03d}'  # Three-digit zero-padded (e.g., '001', '601')
    else:
        return f'{i:04d}' # Four-digit without padding (e.g., '1117', '1423')

# Function to assign folds with exact 7:1:2 ratio
def assign_folds_exact(df, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Total counts
    total_samples = len(df)
    train_count = int(total_samples * train_ratio)
    valid_count = int(total_samples * valid_ratio)
    test_count = total_samples - train_count - valid_count  # Ensure total adds up

    # Split per label
    depression_df = df[df['label'] == 'depression']
    normal_df = df[df['label'] == 'normal']

    # Calculate counts per label
    total_depression = len(depression_df)
    total_normal = len(normal_df)

    # Depression splits
    depression_train_count = int(total_depression * train_ratio)
    depression_valid_count = int(total_depression * valid_ratio)
    depression_test_count = total_depression - depression_train_count - depression_valid_count

    # Normal splits
    normal_train_count = int(total_normal * train_ratio)
    normal_valid_count = int(total_normal * valid_ratio)
    normal_test_count = total_normal - normal_train_count - normal_valid_count

    # Assign folds for depression
    depression_df = shuffle(depression_df, random_state=42)
    depression_train = depression_df.iloc[:depression_train_count].copy()
    depression_valid = depression_df.iloc[depression_train_count:depression_train_count + depression_valid_count].copy()
    depression_test = depression_df.iloc[depression_train_count + depression_valid_count:].copy()

    depression_train['fold'] = 'train'
    depression_valid['fold'] = 'valid'
    depression_test['fold'] = 'test'

    # Assign folds for normal
    normal_df = shuffle(normal_df, random_state=42)
    normal_train = normal_df.iloc[:normal_train_count].copy()
    normal_valid = normal_df.iloc[normal_train_count:normal_train_count + normal_valid_count].copy()
    normal_test = normal_df.iloc[normal_train_count + normal_valid_count:].copy()

    normal_train['fold'] = 'train'
    normal_valid['fold'] = 'valid'
    normal_test['fold'] = 'test'

    # Combine all splits
    result_df = pd.concat([
        depression_train, depression_valid, depression_test,
        normal_train, normal_valid, normal_test
    ], ignore_index=True)

    return result_df

if __name__ == '__main__':

    ''' 
    ## About Dataset

        1.The "Audio_feature" directory stores all extracted audio features.

                Here are the depression category number:
                - 001-601
                - 1117-1423 

                and the normal class here:
                - 0602-1116
                - 1425-1824

        2.The "Video_feature" directory stores all extracted video features,the number is the same as Audio_feature.
    
    '''

    # Define the ranges for depression and normal classes
    depression_indices = list(range(1, 602)) + list(range(1117, 1424))  # 001-601 and 1117-1423
    normal_indices = list(range(602, 1117)) + list(range(1425, 1825))   # 0602-1116 and 1425-1824

    ## Format indices as four-digit strings (e.g., '0001', '1117')
    # depression_indices = [f'{i:04d}' for i in depression_indices]
    # normal_indices = [f'{i:04d}' for i in normal_indices]

    # Format indices
    depression_indices = [format_index(i) for i in depression_indices]
    normal_indices = [format_index(i) for i in normal_indices]

    # Create DataFrames for each class
    depression_df = pd.DataFrame({'index': depression_indices, 'label': 'depression'})
    normal_df = pd.DataFrame({'index': normal_indices, 'label': 'normal'})

    # Combine into one DataFrame
    df = pd.concat([depression_df, normal_df], ignore_index=True)

    # Shuffle the DataFrame
    df = shuffle(df, random_state=42)

    # Assign folds with exact counts
    result_df = assign_folds_exact(df, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2)

    # Save to CSV
    output_file = '../data/lmvd-dataset/lmvd_labels.csv'
    result_df.to_csv(output_file, index=False)
    print(f"CSV file saved as {output_file}")

    # Print the distribution to verify
    print("\nDistribution of folds:")
    print(result_df.groupby(['label', 'fold']).size())

    ''' 
    Distribution of folds:
        label       fold 
        depression  test     183
                    train    635
                    valid     90
        normal      test     184
                    train    640
                    valid     91
        dtype: int64
    '''