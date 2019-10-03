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
import os
import torch.nn as nn
from torchvision.models import resnet50
import apex
from apex import amp, optimizers                                                 
import argparse                                                        

parser = argparse.ArgumentParser()                                               
parser.add_argument('--local_rank', default=0,type=int)                                   
args = parser.parse_args()                                                   

def main():
	args.distributed = False
	if 'WORLD_SIZE' in os.environ:
		args.distributed = int(os.environ['WORLD_SIZE']) > 1

	if args.distributed:
		# FOR DISTRIBUTED:  Set the device according to local_rank.
		torch.cuda.set_device(args.local_rank)

		# FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
		# environment variables, and requires that you use init_method=`env://`.
		torch.distributed.init_process_group(backend='nccl',
											init_method='env://')

	torch.backends.cudnn.benchmark = True

	torch.cuda.set_device(args.local_rank)                                        # FOR APEX PARALLEL
	 
	model = resnet50().cuda()
	x = torch.randn(64, 3, 224, 224).cuda()
	gt = torch.tensor(64 * [1]).long().cuda()
 
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()
 
	model, optimizer = amp.initialize(model, optimizer, opt_level='O1')          # FOR APEX

	if args.distributed:
		# FOR DISTRIBUTED:  After amp.initialize, wrap the model with
		# apex.parallel.DistributedDataParallel.
		model = apex.parallel.DistributedDataParallel(model)                         # FOR APEX PARALLEL
	
	
	for _ in tqdm(range(10000)):
		y = model(x)
		loss = criterion(y, gt)
		with amp.scale_loss(loss, optimizer) as scaled_loss:                     # FOR APEX
			scaled_loss.backward()                                               # FOR APEX
		optimizer.step()


if __name__ =='__main__':
	main()