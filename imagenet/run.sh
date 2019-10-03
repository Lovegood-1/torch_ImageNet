#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 imagenet.py --pretrained --lr=0.01  /data/behavior_data/student_data/lexue
python -m torch.distributed.launch --nproc_per_node=8 imagenet.py \
                                    --pretrained --lr=0.01  /data/behavior_data/student_data/lexue

### 支持两种方式的初始化, 
##### 第一种初始化方式: tcp 初始化
python imagenet_main.py -a resnet18  \
                        --dist-url 'tcp://192.168.68.58:8810' --dist-backend 'nccl' \
                        --multiprocessing-distributed --world-size 1 --rank 0 \
                        --pretrained --lr=0.01 /data/behavior_data/student_data/lexue

##### 第二种初始化方式: env 初始化
python imagenet_main.py -a resnet18  \
                        --dist-url 'env://' --dist-backend 'nccl' \
                        --multiprocessing-distributed --world-size 1 --rank 0 \
                        --pretrained --lr=0.01 /data/behavior_data/student_data/lexue


##### 使用 lunch 的方式启动分布式
python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py -a resnet18  \
                        --dist-url 'tcp://192.168.68.58:8810' --dist-backend 'nccl' \
                        --multiprocessing-distributed --world-size 1 --rank 0 \
                        --pretrained --lr=0.01 /data/behavior_data/student_data/lexue


#### 使用 Apex 混合精度训练
python -m torch.distributed.launch --nproc_per_node=8 imagenet_amp.py -a resnet18 \
                            --workers 4 --opt-level O2  --deterministic --prof  2 \
                            --pretrained --b 32 --lr=0.01 /data/behavior_data/student_data/lexue