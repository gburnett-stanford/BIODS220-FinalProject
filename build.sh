#!/bin/bash 

# INPUT_FOLDER="Dataset_20181026/Eye-View_ImagesForLabeling-Preprocessed"
# OUTPUT_FOLDER="clean_data/eye_view_all"
INPUT_FOLDER="Dataset_20181026/TrainingImages"
OUTPUT_FOLDER="clean_data/hand_view_all_aug"

rm -rf $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER
python build_dataset.py --input_folder ${INPUT_FOLDER} --output_folder ${OUTPUT_FOLDER} --all # --participant_id 1