import numpy as np
import torch
n = 6
m=5
bs=2
nlist = range(1, n)
nlist = np.tile(nlist, (4, 1))
nlist = np.transpose(nlist)

mlist = range(1, m)
m = np.tile(mlist, (5, 1))
M = np.zeros((2, 5, 4))
M[0, :, ] = nlist
M[1, :, :] = m
mm = torch.tensor(M)
mm = mm.unsqueeze(0).repeat(bs, 1,1, 1)

nn = np.zeros((2,5,4))
nn = torch.tensor(nn)
nn = nn.unsqueeze(0).repeat(bs, 1,1, 1)
print(mm.max(dim=1)[1])
e = nn
e[(nn+1)==mm] = 1

print(e)