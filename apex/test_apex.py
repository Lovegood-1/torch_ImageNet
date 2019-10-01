from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import resnet50
from apex import amp, optimizers                                                  # FOR APEX


if __name__ == '__main__':
    model = resnet50().cuda()
    x = torch.randn(64, 3, 224, 224).cuda()
    gt = torch.tensor(64 * [1]).long().cuda()
 
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
 
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')           # FOR APEX
     
    for _ in tqdm(range(10000)):
        y = model(x)
        loss = criterion(y, gt)
        with amp.scale_loss(loss, optimizer) as scaled_loss:                      # FOR APEX
            scaled_loss.backward()                                                # FOR APEX
        optimizer.step()