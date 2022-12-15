#!/bin/bash 

# COMMONG ARGS 
DATA_DIR="clean_data/hand_view_all_aug"
b_size=64
l_rate=1e-4

#########################################################

# INITIAL DROPOUT
# dropout=0.0

# UNFREEZE = none
# unfreeze="none"
# OUT_DIR="training/unfreeze_none"
# rm -rf $OUT_DIR
# mkdir -p $OUT_DIR

# python train.py \
#     --data_dir ${DATA_DIR} \
#     --out_dir ${OUT_DIR} \
#     --batch_size ${b_size} \
#     --learning_rate ${l_rate} \
#     --dropout ${dropout} \
#     --unfreeze "none" |& tee ${OUT_DIR}/train.log

# # UNFREEZE = conv5_block3_3
# unfreeze="conv5_block3_3"
# OUT_DIR="training/unfreeze_conv5_block3_3"
# rm -rf $OUT_DIR
# mkdir -p $OUT_DIR

# python train.py \
#     --data_dir ${DATA_DIR} \
#     --out_dir ${OUT_DIR} \
#     --batch_size ${b_size} \
#     --learning_rate ${l_rate} \
#     --dropout ${dropout} \
#     --unfreeze "conv5_block3_3" |& tee ${OUT_DIR}/train.log

# # UNFREEZE = conv5_block3
# unfreeze="conv5_block3"
# OUT_DIR="training/unfreeze_conv5_block3"
# rm -rf $OUT_DIR
# mkdir -p $OUT_DIR

# python train.py \
#     --data_dir ${DATA_DIR} \
#     --out_dir ${OUT_DIR} \
#     --batch_size ${b_size} \
#     --learning_rate ${l_rate} \
#     --dropout ${dropout} \
#     --unfreeze "conv5_block3" |& tee ${OUT_DIR}/train.log

#########################################################

# FINAL UNFREEZE 
unfreeze="conv5_block3"

# DROPOUT = 0.20
# dropout=0.20
# OUT_DIR="training/dropout_020"
# rm -rf $OUT_DIR
# mkdir -p $OUT_DIR

# python train.py \
#     --data_dir ${DATA_DIR} \
#     --out_dir ${OUT_DIR} \
#     --batch_size ${b_size} \
#     --learning_rate ${l_rate} \
#     --dropout ${dropout} \
#     --unfreeze ${unfreeze} |& tee ${OUT_DIR}/train.log

# # DROPOUT = 0.40
# dropout=0.40
# OUT_DIR="training/dropout_040"
# rm -rf $OUT_DIR
# mkdir -p $OUT_DIR

# python train.py \
#     --data_dir ${DATA_DIR} \
#     --out_dir ${OUT_DIR} \
#     --batch_size ${b_size} \
#     --learning_rate ${l_rate} \
#     --dropout ${dropout} \
#     --unfreeze ${unfreeze} |& tee ${OUT_DIR}/train.log

# # DROPOUT = 0.60
# dropout=0.60
# OUT_DIR="training/dropout_060"
# rm -rf $OUT_DIR
# mkdir -p $OUT_DIR

# python train.py \
#     --data_dir ${DATA_DIR} \
#     --out_dir ${OUT_DIR} \
#     --batch_size ${b_size} \
#     --learning_rate ${l_rate} \
#     --dropout ${dropout} \
#     --unfreeze ${unfreeze} |& tee ${OUT_DIR}/train.log

# DROPOUT = 0.80
dropout=0.80
OUT_DIR="training/dropout_080"
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

python train.py \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUT_DIR} \
    --batch_size ${b_size} \
    --learning_rate ${l_rate} \
    --dropout ${dropout} \
    --unfreeze ${unfreeze} |& tee ${OUT_DIR}/train.log

###################################################

