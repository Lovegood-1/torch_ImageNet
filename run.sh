#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 imagenet.py \
                                    --a resnet18 --b 256  \
                                    --pretrained --lr=0.01  /data/behavior_data/student_data/lexue