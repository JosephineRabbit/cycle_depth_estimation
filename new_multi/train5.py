import sys
import time
from options.train_options import TrainOptions
from my_seg_depth.trymulti.semantic_trans.try_data import dataloader
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import cv2
import shutil
import numpy as np
import os
from torch.nn import init
import torch
#from .model import Seg_Depth
#from  .networks import G_1,Discriminator,Feature_net,SEG,DEP
from model5 import Seg_Depth
import torch
import itertools
from util.image_pool import ImagePool
import torch.nn as nn
from my_eval import eval_metric
from collections import OrderedDict


def create_model_segdepth(opt):
    print(opt.model)
    model = Seg_Depth()
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model
if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset_train = dataloader(opt, train_or_test='train')
    dataset_test = dataloader(opt, train_or_test='test')
    writer_train = SummaryWriter(log_dir='./summary/1_21_vt_t')
    writer_test = SummaryWriter(log_dir='./summary/1_21_vt_t')
    opt = TrainOptions().parse()

    dataset_train = dataloader(opt, train_or_test='train')

    dataset_test = dataloader(opt, train_or_test='test')
    model =create_model_segdepth(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    global_iter=0

    for epoch in range(1,30):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset_train):
            print(global_iter)
            global_iter += 1
            iter_start_time = time.time()

            total_steps += opt.batch_size

            epoch_iter += opt.batch_size


            model.set_input(data,train_or_test='train')
            model.optimize_parameters(train_or_test='train')
            model.visual(train_or_test='train')
            if (global_iter % 20 == 0):
                # errors = model.get_current_losses()
                # for name, error in errors.items():
                #     print('------------------')
                #     writer_train.add_scalar("{}train/{}".format(opt.name, name), error, global_iter)
                images = model.get_current_visuals()

                for name, img in images.items():
                    img = img / img.max()
                    # if len(img.shape)==3:

                    # if (name in segname_syn):
                    print('show_shape', img.shape,'show_name',name)
                    img = torch.from_numpy(img.transpose([2, 0, 1]))
                    writer_train.add_image("{}train/img_{}".format(opt.name, name), img, global_iter)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('iter_%d' % total_steps)

            iter_data_time = time.time()

            if global_iter % 1000 == 0:
                print("validation start")
                model.eval()
                model.visual(train_or_test='test')
                shutil.rmtree('/home/dut-ai/Documents/depth_selection/val_selection_cropped/pred')
                os.mkdir('/home/dut-ai/Documents/depth_selection/val_selection_cropped/pred')
                for ii, data_test in enumerate(dataset_test):

                    with torch.no_grad():
                        model.set_input(data_test,train_or_test='test')
                        model.optimize_parameters(train_or_test='test')

                    images = model.get_eval_visuals()
                    for name, img in images.items():
                        img = img / img.max()
                        print('show_shape',img.shape)
                        # if len(img.shape)==3:
                        img = torch.from_numpy(img.transpose([2, 0, 1]))
                        #writer_train.add_image("{}test/img_{}".format(opt.name, name), img, global_iter + ii)
                        if name=='real_dep_ref':
                            print(str(model.return_name()[0][0])+'groundtruth_depth'+str(model.return_name()[1][0]))
                            cv2.imwrite(
                                # '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/save_kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/dep_ref/'
                                '/home/dut-ai/Documents/depth_selection/val_selection_cropped/pred/'+
                                str(model.return_name()[0][0])+'groundtruth_depth'+str(model.return_name()[1][0]), np.array(img[0,:, :])*255)
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3=eval_metric()
                print('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel', 'sq_rel', 'rmse', 'rmse_log','a1', 'a2', 'a3'))
                print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3))
                with open('records.txt', 'a+') as f:
                    f.write(str(epoch) + "-"+str(global_iter) + '{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'.format(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) + "\n")
                print("validation done")

        #if epoch % 2 == 0:
         #   print('saving the model at the end of epoch %d, iters %d' %
          #        (epoch, total_steps))
           # model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))





