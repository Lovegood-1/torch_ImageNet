# filename: distributed_example.py
# import some module
...
...
 
parser = argparse.Argument()
parser.add_argument('--init_method', defalut='env://', type=str)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse()
 
import os
# Set master information and NIC
# NIC for communication
os.environ['NCCL_SOCKET_IFNAME'] = 'xxxx'
# set master node address
# recommend setting to ib NIC to gain bigger communication bandwidth
os.environ['MASTER_ADDR'] = '192.168.xx.xx'
# set master node port
# **caution**: avoid port conflict
os.environ['MASTER_PORT'] = '1234'
 
def main():
    # step 1
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
     
    # step 2
    # set target device
    torch.cuda.set_device(args.local_rank)
 
 
    # step 3
    # initialize process group
    # use nccl backend to speedup gpu communication
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method)
     
    ...
    ...
    # step 4
    # set distributed sampler
    # the same, you can set distributed sampler for validation set
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS
    )
     
    ...
    ...
    # step 5
    # initialize model 
    model = resnet50()
    model.cuda()
     
    # step 6
    # wrap model with distributeddataparallel
    # map device with model process, and we bind process n with gpu n(n=0,1,2,3...) by default.
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
     
    ...
    ...
    for epoch in range(epochs):
        # step 7
        # update sampler with epoch
        train_sampler.set_epoch(epoch)
         
        # then do whatever you want to do, just like a single device/gpu training program
        ...
        ...