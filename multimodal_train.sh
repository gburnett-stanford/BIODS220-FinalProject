#!/bin/bash 

# for b_size in 32
# do 
#     for l_rate in 1e-2 1e-3 1e-4 1e-5
#     do
#         DATA_DIR="clean_data/eye_view_all/train"
#         OUT_DIR="training_fine_tune/resnet50_eye_all_batch_${b_size}_lr_${l_rate}"
#         rm -rf $OUT_DIR
#         mkdir -p $OUT_DIR
#         python train.py --data_dir ${DATA_DIR} --out_dir ${OUT_DIR} --batch_size ${b_size} --learning_rate ${l_rate} |& tee ${OUT_DIR}/train.log

#         DATA_DIR="clean_data/hand_view_all_aug/train"
#         OUT_DIR="training_fine_tune/resnet50_hand_all_batch_${b_size}_lr_${l_rate}"
#         rm -rf $OUT_DIR
#         mkdir -p $OUT_DIR
#         python train.py --data_dir ${DATA_DIR} --out_dir ${OUT_DIR} --batch_size ${b_size} --learning_rate ${l_rate} |& tee ${OUT_DIR}/train.log
#     done
# done

b_size=64
l_rate=1e-5
EYE_DIR="clean_data/eye_view_all/"
HAND_DIR="clean_data/hand_view_all_aug/"
OUT_DIR="training_fine_tune/test"
rm -rf $OUT_DIR
mkdir -p $OUT_DIR
python multimodal_train.py --hand_dir ${HAND_DIR} --eye_dir ${EYE_DIR} --out_dir ${OUT_DIR} --batch_size ${b_size} --learning_rate ${l_rate} |& tee ${OUT_DIR}/train.log
