import sys
import time
from options.train_options import TrainOptions
from my_seg_depth.trymulti.my_data import dataloader
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
from torch.nn import init
import torch
#from .model import Seg_Depth
#from  .networks import G_1,Discriminator,Feature_net,SEG,DEP
from my_seg_depth.trymulti.model3 import Seg_Depth
import torch
import itertools
from util.image_pool import ImagePool
import torch.nn as nn
from util.util import scale_pyramid
import os
import util.util as util
from collections import OrderedDict


my_weights = [
    1.2,#ground
    0.9,#road
    1.3,#sidewalk
    1.3,#parking
    1.3,#railtrack
    0.9,#building
    1.1,#wall
    1.2,#fence
    1.2,#guardrail
    1.3,#bridge
    1.3,#tunnel
    1.3,#pole
    1.3,#polegroup
    1.4,#trafficlight
    1.4,#trafficsign
    1.2,#vegetation
    1.3,#terrain
    1.1,#sky
    1.5,#person
    1.6,#rider
    1.1,#car
    1.3,#truck
    1.3,#bus
    1.3,#carvan
    1.5,#trailer
    1.5,#train
    1.6,#motorcycle
    1.4,#biicycle
    ]

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
    writer_train = SummaryWriter(log_dir='./summary/12_scale2')
    writer_test = SummaryWriter(log_dir='./summary/12_scale2')
    opt = TrainOptions().parse()
    #opt = TrainOptions().parse()
    dataset_train = dataloader(opt, train_or_test='train')

    #dataset_train = dataloader(opt, train_or_test='train')

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
            #if total_steps % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters(train_or_test='train')
            if (global_iter % 20 == 0):
                errors = model.get_current_losses()
                for name, error in errors.items():
                    print('------------------')
                    writer_train.add_scalar("{}train/{}".format(opt.name, name), error, global_iter)
                images = model.get_current_visuals()

                for name, img in images.items():

                    img = img / img.max()
                   # if len(img.shape)==3:
                    img = torch.from_numpy(img.transpose([2, 0, 1]))
                    writer_train.add_image("{}train/img_{}".format(opt.name, name), img, global_iter)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('iter_%d' % total_steps)

            iter_data_time = time.time()

            if global_iter % 100 == 0:
                print("validation start")
                model.eval()
                for ii, data_test in enumerate(dataset_test):
                    if (ii == 10):
                        break
                    with torch.no_grad():
                        model.set_input(data_test)
                        model.optimize_parameters(train_or_test='test')
                    errors = model.get_current_losses()
                    for name, error in errors.items():
                        writer_train.add_scalar("{}test/{}".format(opt.name, name), error, global_iter + ii)
                    images = model.get_current_visuals()
                    for name, img in images.items():
                        img = img / img.max()
                        # if len(img.shape)==3:
                        img = torch.from_numpy(img.transpose([2, 0, 1]))
                        writer_train.add_image("{}test/img_{}".format(opt.name, name), img, global_iter + ii)
                print("validation done")

        if epoch % 5 == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        #model.update_learning_rate()






