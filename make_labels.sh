#!/bin/bash 

# TEST_DIR="clean_data/eye_view_all/test"
TEST_DIR="clean_data/hand_view_all_aug/test"
python make_labels.py --test_dir ${TEST_DIR}
