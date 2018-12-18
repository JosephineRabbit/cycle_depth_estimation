import torch
import itertools
from util.image_pool import ImagePool
#from .base_model import BaseModel
from my_seg_depth.new_depseg import networks2
from my_seg_depth.new_depseg.networks2 import init_net, init_weights
#from .encoder_decoder import _UNetEncoder,_UNetDecoder
import torch.nn as nn
from util.util import scale_pyramid
import os
import util.util as util
from collections import OrderedDict
import functools
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
class BaseModel():

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self,opt):
        self.opt = opt

        self.is_Train  = opt.isTrain

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
        if self.is_Train:
            self.schedulers = [networks2.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.is_Train or opt.continue_train:
            self.load_networks2(opt.epoch)
        self.print_networks2(opt.verbose)

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
            save_dir = '/checkpoints/4dis'

            print(name)
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(save_dir, save_filename)
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
    def load_networks2(self, epoch):
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
    def print_networks2(self, verbose):
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

my_weights = torch.tensor(
    (
    1.4,#ground 0
    0.8,#road 1
    1.2,#sidewalk 2
    1.4,#parking 3
    1.3,#railtrack 4
    0.8,#building 5
    1.3,#wall 6
    1.4,#fence 7
    1.4,#guardrail 8
    1.4,#bridge 9
    1.4,#tunnel 10
    1.4,#pole 11
    1.4,#polegroup 12
    1.5,#trafficlight 13
    1.5,#trafficsign 14
    1.2,#vegetation 15
    1.3,#terrain 16
    1.1,#sky 17
    2,#person 18
    2,#rider19
    1.1,#car 20
    1.8,#truck
    1.8,#bus
    1.8,#carvan
    1.8,#trailer
    1.8,#train
    1.8,#motorcycle
    1.8,#biicycle
    )
)


class Seg_Depth(BaseModel):

    def name(self):
        return 'Seg_Depth_Model'




    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        self.loss_names=['G2_dis','D_syn','dep_syn',#'DIS_real','DIS_syn',
                         #'seg_syn',
                         'seg_real','D_real',#'adv_DIS_syn',
                          ]
        self.visual_names = ['syn_img','real_img','syn_seg_l','real_seg_l',
                             'syn_seg_pre','real_seg_pre','syn_dep_l','syn_dep_pre','real_dep_pre',
                             ]
        if self.is_Train:
            self.model_names = ['G_1', 'G_2', 'Dis0_en','Dis1_en','Dis2_en','Dis3_en','Dep_de',
                               'Seg_de'#,'DIS'
                                ]
        else:  # during test time, only load Gs
            self.model_names = ['G_1', 'G_2','Dep_de',
                               'Seg_de']
        self.net_Dis0_en = networks2.Discriminator2_seg().cuda()
        self.net_Dis0_en = init_net(self.net_Dis0_en)
        self.net_Dis1_en = networks2.Discriminator2_seg().cuda()
        self.net_Dis1_en = init_net(self.net_Dis1_en)
        self.net_Dis2_en = networks2.Discriminator2_seg().cuda()
        self.net_Dis2_en = init_net(self.net_Dis2_en)
        self.net_Dis3_en = networks2.Discriminator2_seg().cuda()
        self.net_Dis3_en = init_net(self.net_Dis3_en)
       # self.net_Dis_en = nn.DataParallel(self.net_Dis_en)

    #    self.net_Dis_en.load_state_dict(torch.load(
     #       '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/new_depseg/checkpoints/new_seg2dep/'
      #      'iter_60000_net_Dis_en.pth'))

        self.net_G_1 = networks2.General_net().cuda()
        self.net_G_1 = nn.DataParallel(self.net_G_1)


        self.net_G_1.load_state_dict(torch.load(
            '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/new_depseg/checkpoints/new_seg2dep/'
            'iter_60000_net_G_1.pth'))
        print('1')

        self.net_G_2 = networks2.General_net().cuda()
        self.net_G_2 = nn.DataParallel(self.net_G_2)

        self.net_G_2.load_state_dict(torch.load(
            '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/new_depseg/checkpoints/new_seg2dep/'
            'iter_60000_net_G_2.pth'))


        print('2')



        self.net_Seg_de = networks2.SEG(n_cls=28).cuda()
       ## self.net_Seg_de = init_net(self.net_Seg_de)
        self.net_Seg_de = nn.DataParallel(self.net_Seg_de)
        self.net_Seg_de.load_state_dict(torch.load(
           '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/new_depseg/checkpoints/new_seg2dep/'
            'iter_60000_net_Seg_de.pth'))
        #self.net_Seg_de = nn.DataParallel(self.net_Seg_de)
        self.net_Dep_de = networks2.DEP().cuda()
       # self.net_Dep_de = init_net(self.net_Dep_de)
        self.net_Dep_de = nn.DataParallel(self.net_Dep_de)
        self.net_Dep_de.load_state_dict(torch.load(
            './checkpoints/'
            'new_seg2dep/iter_60000_net_Dep_de.pth'
        ))
        #self.net_Dep_de = nn.DataParallel(self.net_Dep_de)


        self.optimizer_G_1 = torch.optim.Adam(self.net_G_1.parameters(),
                                            lr=opt.lr/2, betas=(opt.beta1, 0.999))
        self.optimizer_G_2 = torch.optim.Adam(self.net_G_2.parameters(),
                                              lr=opt.lr/2,betas=(opt.beta1,0.999))

        self.optimizer_Seg = torch.optim.Adam(self.net_Seg_de.parameters(),
                                              lr=opt.lr/2,betas=(opt.beta1,0.999))

        self.optimizer_Dep = torch.optim.Adam(self.net_Dep_de.parameters(),
                                            lr=opt.lr/2, betas=(opt.beta1, 0.999))
        self.optimizer_D0 = torch.optim.SGD(self.net_Dis0_en.parameters(),
                                              lr=opt.lr/3)
        self.optimizer_D1 = torch.optim.SGD(self.net_Dis1_en.parameters(),
                                            lr=opt.lr / 3)
        self.optimizer_D2 = torch.optim.SGD(self.net_Dis2_en.parameters(),
                                            lr=opt.lr / 3)
        self.optimizer_D3 = torch.optim.SGD(self.net_Dis3_en.parameters(),
                                            lr=opt.lr / 3)
      #  self.optimizer_DIS = torch.optim.Adam(itertools.hain(self.net_DIS.parameters()),
       #                                     lr=opt.lr/5, betas=(opt.beta1, 0.999))
        self.syn_imgpool = ImagePool(opt.pool_size)
        self.real_imgpool = ImagePool(opt.pool_size)

        self.criterionGAN = networks2.GANLoss(use_lsgan =True)
        self.criterionSeg = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255).cuda()
        self.criterionDep = torch.nn.L1Loss()

    def detach_list(self,list):
        for i in list:
            i = i.detach()
        return list
    def set_input(self,input):
        self.real_img = input['img_real'].cuda()
        self.syn_img = input['img_syn'].cuda()
        self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
        self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
        self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()







    def calc_gradient_penalty(self,netD, real_data, fake_data):

        interpolates = real_data

        #for  i in range(12):
        alpha = torch.rand(1).cuda()
        interpolates[0,:,:,:] = alpha*real_data[0,:,:,:]+ ((1 - alpha) * fake_data[0,:,:,:])

        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        LAMBDA = 10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def cal_DisL(self,real_features1,label):
        pre_r0 = self.net_Dis0_en(real_features1[:, :256, :, :])
        pre_r1 = self.net_Dis1_en(real_features1[:, 256:512, :, :])
        pre_r2 = self.net_Dis2_en(real_features1[:, 512:768, :, :])
        pre_r3 = self.net_Dis3_en(real_features1[:, 768:1024, :, :])

        loss_D_real = self.criterionGAN(pre_r0, label) + self.criterionGAN(pre_r1, label) + self.criterionGAN(pre_r2,
                                                                                                                 label) + self.criterionGAN(
            pre_r3, label)

        return loss_D_real

    def backward_D(self):
        self.optimizer_D0.zero_grad()
        self.optimizer_D1.zero_grad()
        self.optimizer_D2.zero_grad()
        self.optimizer_D3.zero_grad()
        syn_features1 = self.net_G_1(self.syn_img,'R').detach()
        real_features1 = self.net_G_2(self.real_img, 'R').detach()

        pre_s0 = self.net_Dis0_en(syn_features1[:, :256, :, :])
        pre_s1 = self.net_Dis1_en(syn_features1[:, 256:512, :, :])
        pre_s2 = self.net_Dis2_en(syn_features1[:, 512:768, :, :])
        pre_s3 = self.net_Dis3_en(syn_features1[:, 768:1024, :, :])
        self.loss_D_syn0 = self.criterionGAN(pre_s0, True)
        self.loss_D_syn1 = self.criterionGAN(pre_s1, True)
        self.loss_D_syn2 = self.criterionGAN(pre_s2, True)
        self.loss_D_syn3 = self.criterionGAN(pre_s3, True)
        self.loss_D_syn = (self.loss_D_syn0 + self.loss_D_syn1 + self.loss_D_syn2 + self.loss_D_syn3).detach()

        pre_r0 = self.net_Dis0_en(real_features1[:, :256, :, :])
        pre_r1 = self.net_Dis1_en(real_features1[:, 256:512, :, :])
        pre_r2 = self.net_Dis2_en(real_features1[:, 512:768, :, :])
        pre_r3 = self.net_Dis3_en(real_features1[:, 768:1024, :, :])
        self.loss_D_real0 = self.criterionGAN(pre_r0, False)
        self.loss_D_real1 = self.criterionGAN(pre_r1, False)
        self.loss_D_real2 = self.criterionGAN(pre_r2, False)
        self.loss_D_real3 = self.criterionGAN(pre_r3, False)
        self.loss_D_real = (self.loss_D_real0+self.loss_D_real1+self.loss_D_real2+self.loss_D_real3).detach()

        return self.loss_D_real0+self.loss_D_syn0,self.loss_D_real1+self.loss_D_syn1,self.loss_D_real2+self.loss_D_syn2,self.loss_D_real3+self.loss_D_syn3

    def backward_DIS(self):
        self.set_requires_grad([self.net_Dis_en], True)
        self.set_requires_grad([self.net_G_1, self.net_G_2,
                                self.net_Seg_de,self.net_Dep_de], False)
        pre_S = self.net_D(self.syn_features3.detach())
        self.loss_DIS_syn = self.criterionGAN(pre_S, False)
        pre_R = self.net_DIS(self.real_features3.detach())
        self.loss_DIS_real  = self.criterionGAN(pre_R, True)
        loss_grad = self.calc_gradient_penalty(netD =self.net_D,real_data=self.real_features3.detach(),
                                               fake_data=self.syn_features3.detach())

        return self.loss_DIS_syn + self.loss_DIS_real+loss_grad

    def backward_Seg(self):

        seg_syn_pre = self.net_Seg_de(self.syn_features1.detach())

        seg_real_pre = self.net_Seg_de(self.real_features1.detach())
    #    self.loss_seg_syn = self.criterionSeg(seg_syn_pre,self.syn_seg_l)
        self.loss_seg_real = self.criterionSeg(seg_real_pre,self.real_seg_l)
        self.syn_seg_pre = seg_syn_pre.detach()
        self.real_seg_pre = seg_real_pre.detach()
        #pre_s = self.net_DIS(self.syn_features3)
        #self.loss_adv_DIS_syn = self.criterionGAN(pre_s, True)
        #if self.loss_adv_DIS_syn>1.2:
        #    self.loss_adv_DIS_syn = (self.loss_adv_DIS_syn-1.2)*0.2+1.2
        #pre_r = self.net_Dis_en(self.real_features3, self.real_seg_l.unsqueeze(1).float())
        #self.loss_DIS_real = self.criterionGAN(pre_r, False)

        return 1.3*self.loss_seg_real#+self.loss_seg_syn#+0.05*self.loss_adv_DIS_syn

    def backward_Dep(self):
        dep_syn_pre = self.net_Dep_de(self.syn_features1.detach())

        dep_real_pre = self.net_Dep_de(self.real_features1.detach())
        self.loss_dep_syn =  self.criterionDep(dep_syn_pre,self.syn_dep_l)
        self.syn_dep_pre = dep_syn_pre.detach()
        self.real_dep_pre = dep_real_pre.detach()
        return self.loss_dep_syn

    def backward_G_1(self):
        self.set_requires_grad([self.net_Seg_de,self.net_G_2,self.net_Dep_de], False)
        self.syn_features1 = self.net_G_1(self.syn_img, 'R')
       # pre_s = self.net_Dis_en(self.syn_features1)
       # self.loss_G1_dis = self.criterionGAN(pre_s, True)
       # if self.loss_G1_dis>1:
       #     self.loss_G1_loss = torch.log(self.loss_G1_loss)+1

        seg_syn_pre = self.net_Seg_de(self.syn_features1)
      #  self.syn_features1_ = self.net_G_1(self.syn_img, 'R')
        dep_syn_pre = self.net_Dep_de(self.syn_features1)


        loss_dep = self.criterionDep(dep_syn_pre, self.syn_dep_l)

        loss_seg_syn = self.criterionSeg(seg_syn_pre, self.syn_seg_l)
        #seg_syn_pre, seg_syn_feaetures3 = self.net_Seg_de(syn_f2, syn_inf)

        self.loss_G_1 =  loss_seg_syn+loss_dep


        return self.loss_G_1

    def backward_G_2(self):
        self.set_requires_grad([self.net_Seg_de, self.net_G_1, self.net_Dep_de], False)
        self.set_requires_grad([self.net_Dis3_en,self.net_Dis2_en, self.net_Dis1_en, self.net_Dis0_en], False)
        self.real_features1 = self.net_G_2(self.real_img, 'R')
        self.loss_G2_dis = self.cal_DisL(self.real_features1,True)

        seg_real_pre = self.net_Seg_de(self.real_features1)
        loss_seg = self.criterionSeg(seg_real_pre, self.real_seg_l)
        # seg_syn_pre, seg_syn_feaetures3 = self.net_Seg_de(syn_f2, syn_inf)

        self.loss_G_2 =  self.loss_G2_dis + loss_seg
        self.optimizer_D0.zero_grad()
        self.optimizer_D1.zero_grad()
        self.optimizer_D2.zero_grad()
        self.optimizer_D3.zero_grad()

        return self.loss_G_2

    def optimize_parameters(self,train_or_test):


        if (train_or_test == 'train'):
            self.set_requires_grad([self.net_Dis3_en, self.net_Dis2_en, self.net_Dis1_en, self.net_Dis0_en], True)
            self.set_requires_grad([self.net_G_1, self.net_G_2,
                                    self.net_Seg_de, self.net_Dep_de], False)

            self.optimizer_D0.zero_grad()
            self.optimizer_D1.zero_grad()
            self.optimizer_D2.zero_grad()
            self.optimizer_D3.zero_grad()
            self.loss_D0, self.loss_D1, self.loss_D2, self.loss_D3 = self.backward_D()
            self.loss_D0.backward()
            print('dis update')
            self.optimizer_D0.step()
            self.loss_D2.backward()
            print('dis2 update')
            self.optimizer_D2.step()
            self.loss_D3.backward()
            print('dis3 update')
            self.optimizer_D3.step()
            self.loss_D1.backward()
            print('dis1 update')
            self.optimizer_D1.step()

        self.set_requires_grad([self.net_G_1, self.net_G_2,
                                self.net_Seg_de], True)

        self.set_requires_grad([self.net_Seg_de, self.net_G_2,self.net_Dis3_en, self.net_Dis2_en, self.net_Dis1_en, self.net_Dis0_en], False)
        self.optimizer_G_1.zero_grad()
        self.loss_G1 =self.backward_G_1()

        if (train_or_test == 'train'):
            print('g1_update')
            self.loss_G1.backward()
            self.optimizer_G_1.step()


        self.set_requires_grad(self.net_G_2, True)
        self.optimizer_G_2.zero_grad()

        self.loss_G2 = self.backward_G_2()

        if (train_or_test == 'train'):
            print('g2_update')
            self.loss_G2.backward()
            self.optimizer_G_2.step()

        self.set_requires_grad([self.net_Seg_de, self.net_Dep_de],True)
        self.set_requires_grad([self.net_G_1, self.net_G_2],False)#,self.net_DIS], False)


        #self.forward()
        self.optimizer_Seg.zero_grad()
        self.loss_seg = self.backward_Seg()


        if (train_or_test == 'train'):
            self.loss_seg.backward()
            print('seg update')

            self.optimizer_Seg.step()

        self.optimizer_Dep.zero_grad()
        self.loss_dep= self.backward_Dep()


        if (train_or_test == 'train'):
            self.loss_dep.backward()
            print('-----dep update')

            self.optimizer_Dep.step()


        if (train_or_test == 'train'):
            self.set_requires_grad([self.net_Dis3_en, self.net_Dis2_en, self.net_Dis1_en, self.net_Dis0_en], True)
            self.set_requires_grad([self.net_G_1, self.net_G_2,
                                    self.net_Seg_de, self.net_Dep_de], False)

            self.optimizer_D0.zero_grad()
            self.optimizer_D1.zero_grad()
            self.optimizer_D2.zero_grad()
            self.optimizer_D3.zero_grad()
            self.loss_D0, self.loss_D1, self.loss_D2, self.loss_D3 = self.backward_D()
            self.loss_D0.backward()
            print('dis update')
            self.optimizer_D0.step()
            self.loss_D2.backward()
            print('dis2 update')
            self.optimizer_D2.step()
            self.loss_D3.backward()
            print('dis3 update')
            self.optimizer_D3.step()
            self.loss_D1.backward()
            print('dis1 update')
            self.optimizer_D1.step()







