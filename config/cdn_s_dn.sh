#!/usr/bin/env bash

set -x
EXP_DIR=logs/

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port 29501 \
        --use_env \
        main.py \
        --pretrained params/checkpoint.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 60 \
        --lr_drop 30 \
        --use_nms_filter \
        --batch_size 4
