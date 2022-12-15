#!/bin/bash 

for b_size in 64
do 
    for l_rate in 1e-3 1e-4 1e-5
    do
        # EYE VIEW RESNET50
        DATA_DIR="clean_data/eye_view_all/"
        OUT_DIR="training/eye_view/batch_${b_size}_lr_${l_rate}"
        rm -rf $OUT_DIR
        mkdir -p $OUT_DIR
        python train.py --data_dir ${DATA_DIR} --out_dir ${OUT_DIR} --batch_size ${b_size} --learning_rate ${l_rate} |& tee ${OUT_DIR}/train.log

        # HAND VIEW RESNET50 
        DATA_DIR="clean_data/hand_view_all_aug/"
        OUT_DIR="training/hand_view/batch_${b_size}_lr_${l_rate}"
        rm -rf $OUT_DIR
        mkdir -p $OUT_DIR
        python train.py --data_dir ${DATA_DIR} --out_dir ${OUT_DIR} --batch_size ${b_size} --learning_rate ${l_rate} |& tee ${OUT_DIR}/train.log

        # MULTIMODAL RESNET50
        HAND_DIR="clean_data/eye_view_all/"
        EYE_DIR="clean_data/hand_view_all_aug/"
        OUT_DIR="training/multimodal/batch_${b_size}_lr_${l_rate}"
        rm -rf $OUT_DIR
        mkdir -p $OUT_DIR
        python multimodal_train.py --hand_dir ${HAND_DIR} --eye_dir ${EYE_DIR} --out_dir ${OUT_DIR} --batch_size ${b_size} --learning_rate ${l_rate} |& tee ${OUT_DIR}/train.log
    done
done



