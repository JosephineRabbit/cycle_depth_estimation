import torch
import itertools
from util.image_pool import ImagePool
# from .base_model import BaseModel
from resnet import rf_lw101,segd,seg_gan_loss
from my_seg_depth.trymulti.semantic_trans import networks5_ds
from my_seg_depth.trymulti.semantic_trans.networks5_ds import init_net, init_weights, GramMatrix, get_masks
import torch.nn as nn
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

    def initialize(self, opt):
        self.opt = opt

        self.is_Train = opt.isTrain

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.dep_ref_name = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def setup(self, opt, parser=None):
        if self.is_Train:
            self.schedulers = [networks5_ds.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.is_Train or opt.continue_train:
            self.load_networks5_ds(opt.epoch)
        self.print_networks5_ds(opt.verbose)

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

    def optimize_parameters(self, train_or_test):
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
                    visual_data = value[-1].data
                else:
                    visual_data = value.data
                segname_syn = ['syn_seg_l', 'real_seg_l',
                               'syn_seg_pre8', 'real_seg_pre8',
                               'syn_seg_pre4', 'real_seg_pre4',
                               'syn_seg_pre2', 'real_seg_pre2',
                               'syn_seg_pre2_0', 'real_seg_pre2_0']
                if (name in segname_syn):
                    visual_ret[name] = util.label2im(visual_data)

                else:
                    visual_ret[name] = util.tensor2im(visual_data)
        return visual_ret

    def get_eval_visuals(self):
        visual_ret = OrderedDict()
        for name in self.dep_ref_name:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_data = value[-1].data
                else:
                    visual_data = value.data
                visual_ret[name] = util.tensor2im(visual_data)
        return visual_ret

    def save_current_visuals(self):
        visual_ret = OrderedDict()
        segname_syn = ['syn_seg_l', 'real_seg_l',
                       'syn_seg_pre8', 'real_seg_pre8',
                       'syn_seg_pre4', 'real_seg_pre4',
                       'syn_seg_pre2', 'real_seg_pre2',
                       'syn_seg_pre2_0', 'real_seg_pre2_0']
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_data = value[-1].data
                else:
                    visual_data = value.data


               # print(name,segname_syn,(name in segname_syn))
                if (name in segname_syn):
                    visual_ret[name] = util.label2im(visual_data)
                #    print(name,visual_ret[name].shape)

                else:
                    visual_ret[name] = util.tensor2im(visual_data)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            save_dir = './checkpoints/1_30_new'

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


    # print network information
    def print_networks5_ds(self, verbose):
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


class Seg_Depth(BaseModel):
    def name(self):
        return 'Seg_Depth_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.up2 = nn.Upsample(scale_factor=2)
        self.net_G = rf_lw101().cuda()
        self.net_G.load_state_dict(torch.load('/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/new_model/res101_model.pth'))
        self.net_G = nn.DataParallel(self.net_G)

        self.net_seg8 = segd(up_scale=3,n_cls=28)
        self.net_seg8 = init_net(self.net_seg8)

        self.net_seg4 = segd(up_scale=2, n_cls=28)
        self.net_seg4 = init_net(self.net_seg4)

        self.net_seg2 = segd(up_scale=1, n_cls=28)
        self.net_seg2 = init_net(self.net_seg2)

        self.net_seg2_0 = segd(up_scale=1, n_cls=28)
        self.net_seg2_0 = init_net(self.net_seg2_0)


        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                              lr=opt.lr / 4, betas=(opt.beta1, 0.999))
        self.optimizer_seg8 = torch.optim.Adam(self.net_seg8.parameters(),
                                              lr=opt.lr / 3, betas=(opt.beta1, 0.999))
        self.optimizer_seg4 = torch.optim.Adam(self.net_seg4.parameters(),
                                              lr=opt.lr / 2, betas=(opt.beta1, 0.999))
        self.optimizer_seg2 = torch.optim.Adam(self.net_seg2.parameters(),
                                              lr=opt.lr / 2, betas=(opt.beta1, 0.999))
        self.optimizer_seg2_0= torch.optim.Adam(self.net_seg2_0.parameters(),
                                             lr=opt.lr / 2, betas=(opt.beta1, 0.999))


        self.syn_imgpool = ImagePool(opt.pool_size)
        self.real_imgpool = ImagePool(opt.pool_size)

        self.criterionGAN = seg_gan_loss()
        self.criterionSeg = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255)
        self.criterionDep = torch.nn.L1Loss()
        self.criterionStyle = torch.nn.MSELoss()
        self.criterionEdge = nn.BCELoss()
        self.criterionDep_bce = networks5_ds.BCEDepLoss()

    def visual(self, train_or_test):
        if train_or_test == 'train':
            self.loss_names = [  # 'G2_dis',

                'loss_dep_syn',
                'adv_g8_seg_syn',
                'adv_g4_seg_syn',
                'adv_g2_seg_syn',
                'adv_g2_0_seg_syn',

                'adv_g8_seg_real',
                'adv_g4_seg_real',
                'adv_g2_seg_real',
                'adv_g2_0_seg_real',

                'adv_d8_seg_syn',
                'adv_d4_seg_syn',
                'adv_d2_seg_syn',
                'adv_d2_0_seg_syn',

                'adv_d8_seg_real',
                'adv_d4_seg_real',
                'adv_d2_seg_real',
                'adv_d2_0_seg_real',



            ]
            self.visual_names = ['syn_img', 'real_img', 'syn_seg_l',   'real_seg_l',
                                 'syn_seg_pre8', 'real_seg_pre8',
                                  'syn_seg_pre4', 'real_seg_pre4',
                                 'syn_seg_pre2', 'real_seg_pre2',
                                 'syn_seg_pre2_0', 'real_seg_pre2_0',
                                 'syn_dep_l',
                                  'syn_dep_pre', 'real_dep_pre',

                                 ]
            self.dep_ref_name = ['real_dep_pre']
            self.model_names = ['G', 'seg8','seg4','seg2','seg2_0'  # 'Dis0_en',  # ,'Dis1_en',#'Dis2_en','Dis3_en',

                                ]
        else:  # during test time, only load Gs
            self.loss_names = [  # 'G2_dis',

                 'adv_g8_seg_syn',
                 'adv_g4_seg_syn',
                'adv_g2_seg_syn',
                 'adv_g2_0_seg_syn',
            ]
            self.visual_names = ['syn_img', 'real_img', 'syn_seg_l',# 'real_seg_l',
                                 'syn_seg_pre8', 'real_seg_pre8',
                                 'syn_seg_pre4', 'real_seg_pre4',
                                 'syn_seg_pre2', 'real_seg_pre2',
                                 'syn_seg_pre2_0', 'real_seg_pre2_0',
                                 'syn_dep_l',
                                 'syn_dep_pre', 'real_dep_pre',

                                 ]
            self.dep_ref_name = ['real_dep_pre']
            self.model_names = ['G']

    def detach_list(self, list):
        for i in list:
            i = i.detach()
        return list

    def set_input(self, input, train_or_test):
        self.real_img = input['img_real'].cuda()
        self.syn_img = input['img_syn'].cuda()

        if train_or_test == 'train':
            self.is_Train = True
            self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
            self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
            self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()

            self.real_img_paths = input['img_source_paths']
            self.syn_img_paths = input['img_target_paths']
            self.syn_dep_ls = input['depth_l_s'].float().cuda()
            self.syn_seg_le = input['seg_e_syn'].float().cuda()
            self.real_seg_le = input['seg_e_real'].float().cuda()
        else:
            self.f_name = input['f_name']
            self.l_name = input['l_name']
            self.is_Train = False
            # self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
            self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
            self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()
            self.real_img_paths = input['img_source_paths']
            self.syn_img_paths = input['img_target_paths']
            self.syn_dep_ls = input['depth_l_s'].float().cuda()
            self.syn_seg_le = input['seg_e_syn'].float().cuda()
            # self.real_seg_le = input['seg_e_real'].float().cuda()

    def return_name(self):
        return self.f_name, self.l_name

    def calc_gradient_penalty(self, netD, real_data, fake_data):

        interpolates = real_data

        # for  i in range(12):
        alpha = torch.rand(1).cuda()
        interpolates[0, :, :, :] = alpha * real_data[0, :, :, :] + ((1 - alpha) * fake_data[0, :, :, :])

        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        LAMBDA = 10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty



    def backward_DIS(self):
        self.set_requires_grad([self.net_seg2_0,self.net_seg2,self.net_seg8,self.net_seg4], True)
        self.set_requires_grad([self.net_G], False)




        self.optimizer_seg8.zero_grad()
        real_seg8 = self.net_seg8(self.real_features[0].detach())
        syn_seg8 = self.net_seg8(self.syn_features[0].detach())
        seg8_loss_real = self.criterionGAN(self.up2(real_seg8),
                                           self.real_seg_l, target_is_real=False)
        seg8_loss_syn = self.criterionGAN(self.up2(syn_seg8),
                                           self.syn_seg_l, target_is_real=False)
        D_seg8_loss = seg8_loss_syn + seg8_loss_real
        D_seg8_loss.backward()
        self.optimizer_seg8.step()

        self.optimizer_seg4.zero_grad()
        real_seg4 = self.net_seg4(self.real_features[1].detach())
        syn_seg4 = self.net_seg4(self.syn_features[1].detach())
        seg4_loss_real = self.criterionGAN(self.up2(real_seg4),
                                           self.real_seg_l, target_is_real=False)
        seg4_loss_syn = self.criterionGAN(self.up2(syn_seg4),
                                           self.syn_seg_l, target_is_real=False)
        D_seg4_loss = seg4_loss_syn + seg4_loss_real
        D_seg4_loss.backward()
        self.optimizer_seg4.step()


        self.optimizer_seg2.zero_grad()
        real_seg2 = self.net_seg2(self.real_features[2].detach())
        syn_seg2 = self.net_seg2(self.syn_features[2].detach())
        seg2_loss_real = self.criterionGAN(self.up2(real_seg2),
                                           self.real_seg_l, target_is_real=False)
        seg2_loss_syn = self.criterionGAN(self.up2(syn_seg2),
                                           self.syn_seg_l, target_is_real=False)
        D_seg2_loss = seg2_loss_syn + seg2_loss_real
        D_seg2_loss.backward()
        self.optimizer_seg2.step()


        self.optimizer_seg2_0.zero_grad()
        real_seg2_0 = self.net_seg2_0(self.real_features[3].detach())
        syn_seg2_0 = self.net_seg2_0(self.syn_features[3].detach())
        seg2_0_loss_real = self.criterionGAN(self.up2(real_seg2_0),
                                             self.real_seg_l, target_is_real=False)
        seg2_0_loss_syn = self.criterionGAN(self.up2(syn_seg2_0),
                                             self.syn_seg_l, target_is_real=False)
        D_seg2_0_loss = seg2_0_loss_syn+seg2_0_loss_real
        D_seg2_0_loss.backward()
        self.optimizer_seg2_0.step()

        self.adv_d8_seg_syn = seg8_loss_syn.detach()
        self.adv_d4_seg_syn = seg4_loss_syn.detach()
        self.adv_d2_seg_syn = seg2_loss_syn.detach()
        self.adv_d2_0_seg_syn = seg2_0_loss_syn.detach()
        self.adv_d8_seg_real = seg8_loss_real.detach()
        self.adv_d4_seg_real = seg4_loss_real.detach()
        self.adv_d2_seg_real = seg2_loss_real.detach()
        self.adv_d2_0_seg_real = seg2_0_loss_real.detach()

        return



    def test_return(self):
        return self.real_img, self.real_dep_pre#,self.real_seg_l

    def backward_G(self,train_or_test):
        self.optimizer_G.zero_grad()
        real_dep_outs,real_pred_d,real_features = self.net_G(self.real_img)
        real_seg8 = self.net_seg8(real_features[0])
        real_seg4 = self.net_seg4(real_features[1])
        real_seg2 = self.net_seg2(real_features[2])
        real_seg2_0 = self.net_seg2_0(real_features[3])

       # print(real_seg8.shape,real_seg4.shape,real_seg2.shape,real_seg2_0.shape)

        dep_loss_real = 0
        seg8_loss_real = 0
        seg4_loss_real = 0
        seg2_loss_real = 0
        seg2_0_loss_real = 0
        if train_or_test=='train':

            seg8_loss_real = self.criterionGAN(self.up2(real_seg8),
                                                              self.real_seg_l,target_is_real=True)

            seg4_loss_real = self.criterionGAN(self.up2(real_seg4),
                                                              self.real_seg_l,target_is_real=True)
            seg2_loss_real =  self.criterionGAN(self.up2(real_seg2),
                                                              self.real_seg_l, target_is_real=True)
            seg2_0_loss_real =  self.criterionGAN(self.up2(real_seg2_0),
                                                              self.real_seg_l, target_is_real=True)
           # print(float(seg8_loss_real), float(seg4_loss_real))
            self.adv_g8_seg_real = seg8_loss_real.detach()
            self.adv_g4_seg_real = seg4_loss_real.detach()
            self.adv_g2_seg_real = seg2_loss_real.detach()
            self.adv_g2_0_seg_real = seg2_0_loss_real.detach()

        G_real_loss = dep_loss_real + seg8_loss_real + seg4_loss_real + seg2_0_loss_real + seg2_loss_real

        self.real_dep_pre = self.up2(real_pred_d).detach()

        syn_dep_outs, syn_pred_d, syn_features = self.net_G(self.syn_img)
        syn_seg8 = self.net_seg8(syn_features[0])
        syn_seg4 = self.net_seg4(syn_features[1])
        syn_seg2 = self.net_seg2(syn_features[2])
        syn_seg2_0 = self.net_seg2_0(syn_features[3])

#        segs_loss_syn = []
        dep_loss_syn = 0
        seg8_loss_syn = 0
        seg4_loss_syn = 0
        seg2_loss_syn = 0
        seg2_0_loss_syn = 0
        if train_or_test=='train':
            seg8_loss_syn = self.criterionGAN(self.up2(syn_seg8),
                                               self.syn_seg_l, target_is_real=True)

            seg4_loss_syn = self.criterionGAN(self.up2(syn_seg4),
                                               self.syn_seg_l, target_is_real=True)
            seg2_loss_syn = self.criterionGAN(self.up2(syn_seg2),
                                                               self.syn_seg_l, target_is_real=True)
            seg2_0_loss_syn =  self.criterionGAN(self.up2(syn_seg2_0),
                                                                 self.syn_seg_l, target_is_real=True)


            sky_m = self.syn_seg_l.clone()
            sky_m[sky_m != 17] = 1
            sky_m[sky_m == 17] = 0
            #print(sky_m.shape, 'sky_m')
            oms, zms = get_masks(torch.cat([sky_m.unsqueeze(1), sky_m.unsqueeze(1),
                                            sky_m.unsqueeze(1), sky_m.unsqueeze(1)],
                                           1).float() * self.syn_dep_ls.clone())

            syn_pred_d = self.up2(syn_pred_d)
            dep_loss_syn = 20 * self.criterionDep(sky_m.float()*syn_pred_d[:, 0, :, :], sky_m.float() * self.syn_dep_l)

            for s_Dep in syn_dep_outs:
                s_Dep = self.up2(s_Dep)
                dep_loss_syn = dep_loss_syn + self.criterionDep_bce(sky_m.unsqueeze(1).float() * s_Dep,
                                                            torch.cat([sky_m.unsqueeze(1), sky_m.unsqueeze(1),
                                                                       sky_m.unsqueeze(1), sky_m.unsqueeze(1)],
                                                                      1).float() * self.syn_dep_ls.clone(), oms, zms)

            self.loss_dep_syn = dep_loss_syn.detach()
            self.adv_g8_seg_syn = seg8_loss_syn.detach()
            self.adv_g4_seg_syn = seg4_loss_syn.detach()
            self.adv_g2_seg_syn = seg2_loss_syn.detach()
            self.adv_g2_0_seg_syn = seg2_0_loss_syn.detach()



        G_syn_loss = dep_loss_syn+seg8_loss_syn+seg4_loss_syn+seg2_0_loss_syn+seg2_loss_syn



        self.syn_dep_pre = syn_pred_d.detach()

        self.syn_seg_pre8 = syn_seg8.detach()
        self.syn_seg_pre4 = syn_seg4.detach()
        self.syn_seg_pre2 = syn_seg2.detach()
        self.syn_seg_pre2_0 = syn_seg2_0.detach()

        self.real_seg_pre8 = real_seg8.detach()
        self.real_seg_pre4 = real_seg4.detach()
        self.real_seg_pre2 = real_seg2.detach()
        self.real_seg_pre2_0 = real_seg2_0.detach()

        self.real_features = self.detach_list(real_features)
        self.syn_features = self.detach_list(syn_features)



        return G_syn_loss+ G_real_loss # +#D_real_loss+D_syn_loss#+D_syn_loss#+10**-30*(loss_style_0+loss_style_1+loss_style_2)





    def optimize_parameters(self, train_or_test):

        self.set_requires_grad([self.net_G], True)

        self.set_requires_grad([self.net_seg2,self.net_seg4,self.net_seg8,self.net_seg2_0], False)
        self.optimizer_G.zero_grad()
        self.loss_G= self.backward_G(train_or_test=train_or_test)

        if (train_or_test == 'train'):
            print('g_update')
            self.loss_G.backward()
            self.optimizer_G.step()


        if (train_or_test == 'train'):
            self.set_requires_grad([self.net_G],False)

            self.set_requires_grad([self.net_seg2, self.net_seg4, self.net_seg8, self.net_seg2_0], True)


            self.backward_DIS()









