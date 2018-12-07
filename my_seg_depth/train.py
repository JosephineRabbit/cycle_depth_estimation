import sys
import time
from options.train_options import TrainOptions
from my_seg_depth.my_data import dataloader
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import my_seg_depth.networks as networks
from torch.nn import init
import torch
#from .model import Seg_Depth
#from  .networks import G_1,Discriminator,Feature_net,SEG,DEP

import torch
import itertools
from util.image_pool import ImagePool
import torch.nn as nn
from util.util import scale_pyramid
import os
import util.util as util
from collections import OrderedDict


class BaseModel():

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self,opt):
        self.opt = opt
#        self.gpu_ids = opt.gpu_ids
        self.is_Train  = opt.isTrain
      #  self.device = torch.device('cuda:{}'.format(self.gpu_ids[0,1])) \
       #     if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self,input):
        self.input = input

    def forward(self):
        pass

    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)

        # make models eval mode during test time

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self,train_or_test):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_data=value[-1].data
                else:
                  visual_data=value.data
                segname_syn=['syn_seg_l',
                             'syn_seg_pre']
                segname_real=['real_seg_l',
                             'real_seg_pre']
                if (name in segname_syn):
                    visual_ret[name]=util.label2im(visual_data)
                elif (name in segname_real):
                    visual_ret[name]=util.label2im(visual_data)
                else:
                    visual_ret[name] = util.tensor2im(visual_data)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self,'loss_'+name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)


                
                torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)

    net = net.cuda()
    net = nn.DataParallel(net)
    init_weights(net, init_type, gain=init_gain)

    return net


class Seg_Depth(BaseModel):

    def name(self):
        return 'Seg_Depth_Model'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)

        self.loss_names=['G1_dis','G1_seg','D_G1',
                         'G2_dis','G2_seg','D_G2',
                         'seg_syn','seg_real','dep_syn'
                         ]

        self.visual_names = ['syn_img','real_img','syn_seg_l','real_seg_l',
                             'syn_seg_pre','real_seg_pre',
                             'syn_dep_l','syn_dep_pre']

        if self.is_Train:
            self.model_names = ['G_1', 'G_2', 'Dis_en',
                                'Feature','Seg_de','Dep_de'
                                ]
        else:  # during test time, only load Gs
            self.model_names = ['G_1', 'G_2',
                                'Feature','Seg_de','Dep_de']

        self.net_G_1 = networks.define_G(opt.input_nc, opt.output_nc,  opt.netG, opt.init_type, opt.init_gain)
        self.net_G_2 = networks.define_G(opt.input_nc, opt.output_nc,  opt.netG,
                                         opt.init_type, opt.init_gain)
        self.net_Dis_en = networks.define_D(input_nc=128)
       # self.net_Feature = networks.Feature_net(input_nc=128,mid_nc =1024)
        self.net_Feature = init_net(networks.Feature_net(input_nc=128,mid_nc =1024))

        self.net_Seg_de = init_net(networks.SEG(n_cls=28))

        self.net_Dep_de = init_net(networks.DEP())


        self.optimizer_G_1 = torch.optim.Adam(self.net_G_1.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G_2 = torch.optim.Adam(self.net_G_2.parameters(),
                                              lr=opt.lr,betas=(opt.beta1,0.999))
        self.optimizer_F = torch.optim.Adam(self.net_Feature.parameters(),
                                            lr=opt.lr,betas=(opt.beta1,0.999))
        self.optimizer_Seg = torch.optim.Adam(self.net_Seg_de.parameters(),
                                              lr=opt.lr,betas=(opt.beta1,0.999))
        self.optimizer_Dep = torch.optim.Adam(self.net_Dep_de.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_Dis_en.parameters()),
                                            lr=opt.lr_D, betas=(opt.beta1, 0.999))


        self.syn_imgpool = ImagePool(opt.pool_size)
        self.real_imgpool = ImagePool(opt.pool_size)

        self.criterionGAN = networks.GANLoss(use_lsgan =not opt.no_lsgan).cuda()
        self.criterionSeg = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255).cuda()
        self.criterionDep = torch.nn.L1Loss()

    def set_input(self,input):
        self.real_img = input['img_real'].cuda()
        self.syn_img = input['img_syn'].cuda()
        self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
        self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
        #print(self.syn_seg_l.shape)
        self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()

    def forward(self):
        self.syn_features1 = self.net_G_1(self.syn_img)
        self.real_features1 = self.net_G_2(self.real_img)

    def forward_features2(self,features1):
        self.features2 = self.net_Feature(features1.detach())
        return self.features2



    def backward_D(self):
        pre_s = self.net_Dis_en(self.syn_features1.detach())
        self.loss_D_G1 = self.criterionGAN(pre_s,False)
        pre_r = self.net_Dis_en(self.real_features1.detach())
        self.loss_D_G2 = self.criterionGAN(pre_r,True)
        return self.loss_D_G1+self.loss_D_G2

    def detach_list(self,list):
        for i in list:
            i = i.detach()
        return list

    def backward_Seg(self):
        syn_f2,syn_inf = self.net_Feature(self.syn_features1.detach())
        real_f2, real_inf = self.net_Feature(self.real_features1.detach())
        self.syn_seg_pre = self.net_Seg_de(self.detach_list(syn_f2),syn_inf.detach())
        self.real_seg_pre = self.net_Seg_de(self.detach_list(real_f2), real_inf.detach())

        self.loss_seg_syn = self.criterionSeg(self.syn_seg_pre,self.syn_seg_l)
        self.loss_seg_real = self.criterionSeg(self.real_seg_pre,self.real_seg_l)

        return self.loss_seg_real+self.loss_seg_syn

    def backward_Dep(self):
        syn_f2, syn_inf = self.net_Feature(self.syn_features1.detach())
        self.syn_dep_pre = self.net_Dep_de(self.detach_list(syn_f2),syn_inf.detach())
        self.loss_dep_syn =  self.criterionDep(self.syn_dep_pre,self.syn_dep_l)
        return self.loss_dep_syn

    def backward_G(self):
        #print(self.syn_features1.shape)
        pre_s = self.net_Dis_en(self.syn_features1)
        self.loss_G1_dis = self.criterionGAN(pre_s, True)
        pre_r = self.net_Dis_en(self.real_features1)
        self.loss_G2_dis = self.criterionGAN(pre_r, False)
        syn_f2, syn_inf = self.net_Feature(self.syn_features1)
        real_f2, real_inf = self.net_Feature(self.real_features1)
        seg_syn_pre = self.net_Seg_de(syn_f2, syn_inf)
        seg_real_pre = self.net_Seg_de(real_f2, real_inf)
        #print(seg_real_pre.shape,self.real_seg_l.shape)
        self.loss_G1_seg = self.criterionSeg(seg_syn_pre, self.syn_seg_l)
        self.loss_G2_seg = self.criterionSeg(seg_real_pre, self.real_seg_l)
       # self.loss_G_1 = self.+loss_seg_real
       # self.loss_G_2 = loss_G_syn+loss_seg_syn

        return self.loss_G1_dis+self.loss_G1_seg,self.loss_G2_dis+self.loss_G2_seg

    def optimize_parameters(self,train_or_test):

        self.set_requires_grad(self.net_Dis_en, False)
        self.set_requires_grad([self.net_G_1, self.net_G_2,
                                self.net_Seg_de, self.net_Feature, self.net_Dep_de], False)
        if (train_or_test=='train'):
            print('train')
            self.set_requires_grad(self.net_Dis_en,False)
            self.set_requires_grad([self.net_G_1,self.net_G_2,
                                    self.net_Seg_de,self.net_Feature,self.net_Dep_de],True)

        self.forward()
        self.optimizer_G_1.zero_grad()
        self.optimizer_G_2.zero_grad()
        self.loss_G1,self.loss_G2 =self.backward_G()

        if (train_or_test == 'train'):
            print('g_update')
            self.loss_G1.backward()
            self.loss_G2.backward()

            self.optimizer_G_1.step()
            self.optimizer_G_2.step()

        self.optimizer_F.zero_grad()
        self.loss_s = self.backward_Seg()
        self.loss_d = self.backward_Dep()
        self.loss_ff = 5*self.loss_s+self.loss_d


        if (train_or_test == 'train'):
            print('F udate')
            self.loss_ff.backward()
            self.optimizer_F.step()
        #self.forward()
        self.optimizer_Seg.zero_grad()
        self.loss_s = self.backward_Seg()


        if (train_or_test == 'train'):
            self.loss_s.backward()
            print('seg update')

            self.optimizer_Seg.step()
        #self.forward()
        self.optimizer_Dep.zero_grad()
        self.loss_d = self.backward_Dep()

        if (train_or_test == 'train'):
            self.loss_d.backward()
            print('dep update')


            self.optimizer_Dep.step()

        if (train_or_test == 'train'):
            self.set_requires_grad(self.net_Dis_en, True)
            self.set_requires_grad([self.net_G_1, self.net_G_2,
                                    self.net_Seg_de, self.net_Feature, self.net_Dep_de], False)

            self.optimizer_D.zero_grad()
            self.loss_D = self.backward_D()
            self.loss_D.backward()
            print('dis update')
            self.optimizer_D.step()


def create_model_segdepth(opt):
    print(opt.model)
    model = Seg_Depth()
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model
if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.name = 'synthia_segCycle'

    dataset_train = dataloader(opt, train_or_test='train')
    dataset_test = dataloader(opt, train_or_test='test')
    writer_train = SummaryWriter(log_dir='./summary/my_seg_depth')
    writer_test = SummaryWriter(log_dir='./summary/my_seg_depth')
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
            if (global_iter % 200 == 0 and global_iter > 200):
                print("validation start")
                model.eval()
                for ii, data_test in enumerate(dataset_test):
                    if (ii == 50):
                        break
                    with torch.no_grad():
                        model.set_input(data_test)
                        model.optimize_parameters(train_or_test='test')
                    errors = model.get_current_losses()
                    for name, error in errors.items():
                        writer_test.add_scalar("{}test/{}".format(opt.name, name), error, global_iter + ii)
                    images = model.get_current_visuals()
                    for name, img in images.items():
                        im = torch.from_numpy(img.transpose([2, 0, 1])).squeeze(0)
                        writer_test.add_image("{}test/img_{}".format(opt.name, name), im, global_iter + ii)
                print("validation done")

            global_iter += 1
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters(train_or_test='train')
            if (global_iter % 50 == 0):
                errors = model.get_current_losses()
                for name, error in errors.items():
                    print('------------------')
                    writer_train.add_scalar("{}train/{}".format(opt.name, name), error, global_iter)
                images = model.get_current_visuals()

                for name, img in images.items():
                    print(name,img.shape)
                    img = img / img.max()
                   # if len(img.shape)==3:
                    img = torch.from_numpy(img.transpose([2, 0, 1]))
                    writer_train.add_image("{}train/img_{}".format(opt.name, name), img, global_iter)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('iter_%d' % total_steps)

            iter_data_time = time.time()

        if epoch % 5 == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()






