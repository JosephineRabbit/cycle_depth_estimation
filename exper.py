import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transform
import numpy as np

weights=torch.load('/home/gwl/PycharmProjects/cloned/pytorch-CycleGAN-and-pix2pix/checkpoints/cycle_gan_synthia/iter_40000_net_G_A.pth')
for key,value in weights.items():
    print(key)