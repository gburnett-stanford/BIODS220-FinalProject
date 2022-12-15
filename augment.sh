#!/bin/bash 

INPUT_FOLDER="Dataset_20181026/Eye-View_ImagesForLabeling"
OUTPUT_FOLDER="Dataset_20181026/Eye-View_ImagesForLabeling-Preprocessed"

rm -rf $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER
python augment.py --input_folder ${INPUT_FOLDER} --output_folder ${OUTPUT_FOLDER} 