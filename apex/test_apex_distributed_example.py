# 从命令行参数读取进程编号
# 多进程调用时会传入一个命令行参数local_rank，代表进程号
"""
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)                              
opt = parser.parse_args()
"""

# 设置多进程组与cuda设备信息
# 注意，此句要放在model进入cuda之前，也就是model.cuda()之前
"""torch.cuda.set_device(opt.local_rank)                   # 设置该进程模型要分配到的gpu编号
"""


# 设置多进程后端
# 注意，此句要放在amp.initialize()之前
"""
torch.distributed.init_process_group(backend='nccl')    # 设置多进程后端为nccl
"""

# 对model进行分布式数据并行包装
# 注意，此句要放在amp.initialize()之后
"""
model = apex.parallel.DistributedDataParallel(model)
"""

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import resnet50
 
from apex import amp, optimizers                                                 # FOR APEX
import apex, argparse                                                            # FOR APEX PARALLEL
parser = argparse.ArgumentParser()                                               # FOR APEX PARALLEL
parser.add_argument('--local_rank', type=int)                                    # FOR APEX PARALLEL
opt = parser.parse_args()                                                        # FOR APEX PARALLEL


if __name__ == '__main__':
    torch.cuda.set_device(opt.local_rank)                                        # FOR APEX PARALLEL
     
    model = resnet50().cuda()
    x = torch.randn(64, 3, 224, 224).cuda()
    gt = torch.tensor(64 * [1]).long().cuda()
 
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
 
    torch.distributed.init_process_group(backend='nccl')                         # FOR APEX PARALLEL
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')          # FOR APEX
    model = apex.parallel.DistributedDataParallel(model)                         # FOR APEX PARALLEL
    for _ in tqdm(range(10000)):
        y = model(x)
        loss = criterion(y, gt)
        with amp.scale_loss(loss, optimizer) as scaled_loss:                     # FOR APEX
            scaled_loss.backward()                                               # FOR APEX
        optimizer.step()