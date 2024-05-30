#!/usr/bin/env bash

set -x
EXP_DIR=logs//

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port 29501 \
        --use_env \
        main.py \
        --pretrained params/checkpoint.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 60 \
        --lr_drop 40 \
        --use_nms_filter \
        --batch_size 4
