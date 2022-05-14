from sortloss2V2_euclidean_all import SortLoss2V2 as sv2
# from sortloss2 import SortLoss2 as sv
import time

# loss1 = sv()
loss2 = sv2()

import torch

a=torch.rand((4,2048))#.cuda()
b=torch.rand((4,2048))#.cuda()

# start = time.time()
# print(loss1(a,b), 'loss1')
# torch.cuda.synchronize()
# print(time.time() - start, '\n')   

start = time.time()
print(loss2(a,b), 'loss2')
torch.cuda.synchronize()
print(time.time() - start)   
