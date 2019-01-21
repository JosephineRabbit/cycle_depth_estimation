import sys
import time
from options.train_options import TrainOptions
from my_seg_depth.trymulti.semantic_trans.try_data import dataloader
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import cv2
from torch.nn import init
import torch
# from .model import Seg_Depth
# from  .networks import G_1,Discriminator,Feature_net,SEG,DEP
from my_seg_depth.trymulti.semantic_trans.test5 import Seg_Depth
import torch
import itertools
from util.image_pool import ImagePool
import torch.nn as nn
#from util.util import scale_pyramid
import os
import util.util as util
from collections import OrderedDict



def create_model_segdepth(opt):
    print(opt.model)
    model = Seg_Depth()
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model


if __name__ == '__main__':
    opt = TrainOptions().parse()
    import numpy as np

    dataset_test = dataloader(opt, train_or_test='test')
    # writer_train = SummaryWriter(log_dir='./summary/19_1_4')
    # writer_test = SummaryWriter(log_dir='./summary/19_1_4')
    opt = TrainOptions().parse()
    # opt = TrainOptions().parse()


    model = create_model_segdepth(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    global_iter = 0

    print("validation start")
    model.eval()
    i = 0
    for ii, data_test in enumerate(dataset_test):



        with torch.no_grad():
            model.set_input(data_test,train_or_test='test')
            model.optimize_parameters(train_or_test='test')
            # errors = model.get_current_losses()
        # for name, error in errors.items():
        #    writer_train.add_scalar("{}test/{}".format(opt.name, name), error, global_iter + ii)
            real_img, real_dep_ref = model.test_return()
            l = real_dep_ref.shape[0]

            for j in range(l):
                i+=1
                print(real_img.shape, real_dep_ref.shape)

                # #$cv2.imwrite(
                #   #  '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/'
                #     'semantic_trans/save_kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'+str(i)+'_img.png',
                #     np.array(np.transpose(real_img[j,:,:,:].cpu(),[1,2,0])* 255))
                cv2.imwrite(
                    #'/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/save_kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/dep_ref/'
                    '/home/dut-ai/Documents/depth_selection/test_depth_completion_anonymous/ref/'
                    + str(model.return_name()[0]), (np.array(real_dep_ref[0, :, :])))
    # img = torch.from_numpy(img.transpose([2, 0, 1]))
           # print("img_{}_{}".format(name, str(ii)), img.shape)

            #     writer_train.add_image("{}test/img_{}".format(opt.name, name), img, global_iter + ii)

    print("validation done")








