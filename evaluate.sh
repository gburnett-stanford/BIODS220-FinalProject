#!/bin/bash 

echo "HAND MODEL"
test_dir="clean_data/eye_view_all/test"
model_path="training_save/hand_view/batch_64_lr_1e-3"
python evaluate.py --test_dir ${test_dir} --model_path ${model_path} 

echo "EYE MODEL"
test_dir="clean_data/eye_view_all/test"
model_path="training_save/eye_view/batch_64_lr_1e-5"
python evaluate.py --test_dir ${test_dir} --model_path ${model_path} 

echo "FUSION MODEL"
test_dir="clean_data/eye_view_all/test"
model_path="training_save/multimodal/batch_64_lr_1e-5"
python evaluate.py --test_dir ${test_dir} --model_path ${model_path} 
