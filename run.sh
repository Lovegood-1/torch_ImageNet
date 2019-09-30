#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 imagenet.py \
                                    --pretrained --lr=0.01  /data/behavior_data/student_data/lexue