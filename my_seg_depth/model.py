import time
from options.train_options import TrainOptions
from datasets.dataset_synthia import dataloader
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import torch


if __name__=='__main__':
    writer_train = SummaryWriter(log_dir='./summary/my_seg_depth')
    writer_test = SummaryWriter(log_dir='./summary/my_seg_depth')
    opt = TrainOptions().parse()
    opt.name = 'my_seg_depth'

    dataset_train = dataloader(opt,train_or_test='train')
    dataset_test = dataloader(opt,train_or_test='test')
    model =
