#!/bin/sh
## 启动程序
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 test_apex_distributed_example.py

python -m torch.distributed.launch --nproc_per_node=8 test_apex_distributed_example.py