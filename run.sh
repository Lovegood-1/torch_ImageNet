#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 imagenet.py --pretrained --lr=0.01  /data/behavior_data/student_data/lexue

python -m torch.distributed.launch --nproc_per_node=8 imagenet.py \
                                    --pretrained --lr=0.01  /data/behavior_data/student_data/lexue