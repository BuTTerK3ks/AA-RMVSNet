#!/bin/bash
source env.sh
data=$(date +"%m%d")
batch=1
epochs=10
d=128
interval_scale=1.06 #make sure d * interval_scale = 203.52
lr=0.001
inverse_depth=False
image_scale=0.25
view_num=5
evidential=True

CUDA_VISIBLE_DEVICES=0 python train.py  \
        --dataset=dtu_yao \
        --batch_size=${batch} \
        --trainpath=$MVS_TRAINING \
        --lr=${lr} \
        --epochs=${epochs} \
        --view_num=$view_num \
        --inverse_depth=${inverse_depth} \
        --image_scale=$image_scale \
        --trainlist=lists/dtu/train.txt \
        --vallist=lists/dtu/val.txt \
        --testlist=lists/dtu/test.txt \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --logdir=./checkpoints/$data \
        --evidential=evidential
