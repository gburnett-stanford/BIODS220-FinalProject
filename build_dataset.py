# This script converts image data from the HANDS dataset Raw Data folder
# into a format useable by TensorFlow 

import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd 

def main(input_folder, output_folder, all, participant_id, shuffle):

    # Load labels 
    train_labels_path = "Dataset_20181026/Labels/Train_Label.csv"
    test_labels_path = "Dataset_20181026/Labels/Test_Label.csv"

    train_labels_df = pd.read_csv(train_labels_path)
    test_labels_df = pd.read_csv(test_labels_path)

    # Downsample for specific participants   
    if all == False: 
        print("Downsampling for participant", participant_id)
        train_labels_df = train_labels_df[train_labels_df["participant_id"]==participant_id]
        test_labels_df = test_labels_df[test_labels_df["participant_id"]==participant_id]
    else: 
        print("Gathering data for all participants")

    # Shuffle
    test_split = 0.2
    if shuffle: 
        print("Shuffling and splitting dataset")
        all_labels = pd.concat([train_labels_df, test_labels_df], ignore_index=True)
        all_labels = all_labels.sample(frac=1)
        nrows = len(all_labels) 
        train_size = int(nrows * (1 - test_split))
        train_labels_df = all_labels.iloc[:train_size]
        test_labels_df = all_labels.iloc[train_size:]

    # Make empty directory structure 
    # labels = ["open_palm", "medium_wrap", "power_sphere", "parallel_extension", "palmar_pinch"]
    labels = ["0", "1", "2", "3", "4"]

    for l in labels: 

        # Make train folder
        label_dir = os.path.join(output_folder, 'train', l)
        os.makedirs(label_dir, exist_ok=True)

        # Make test folder 
        label_dir = os.path.join(output_folder, 'test', l)
        os.makedirs(label_dir, exist_ok=True)

    # Copy training images to their proper directory 
    print("Organizing Training Images")
    for i, row in train_labels_df.iterrows(): 

        # Print progress 
        if i%50==0:
            print("Processing row", i, "of", train_labels_df.shape[0])

        # Get the label 
        # label = labels[row["label_rank1"]] 
        label = str(row["label_rank1"])

        # Get the input image path  
        input_path = os.path.join(input_folder, row["image_name"])

        # Create output image path 
        output_path = os.path.join(output_folder, 'train', label, os.path.basename(input_path))

        # Copy the file 
        if os.path.isfile(input_path):
            shutil.copy(input_path, output_path)
        else: 
            print('Image does not exist:', input_path)

    # Copy training images to their proper directory 
    print("Organizing Test Images")
    for i, row in test_labels_df.iterrows(): 

        # Print progress 
        if i%50==0:
            print("Processing row", i, "of", test_labels_df.shape[0])

        # Get the label 
        # label = labels[row["label_rank1"]] 
        label = str(row["label_rank1"])

        # Get the input image path  
        input_path = os.path.join(input_folder, row["image_name"])

        # Create output image path 
        output_path = os.path.join(output_folder, 'test', label, os.path.basename(input_path))

        # Copy the file 
        if os.path.isfile(input_path):
            shutil.copy(input_path, output_path)
        else: 
            print('Image does not exist:', input_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_folder', type=str, help='input folder of images', required=True)
    parser.add_argument('--output_folder', type=str, help='output folder of images', required=True)
    parser.add_argument('--all', action='store_true', help='whether to use all the participants or just one')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data before splitting')
    parser.add_argument('--participant_id', type=int, default=0, help='id for the patient whose labels are used')

    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.all, args.participant_id, args.shuffle)