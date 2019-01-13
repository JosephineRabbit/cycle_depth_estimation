import torch
import itertools
from util.image_pool import ImagePool
#from .base_model import BaseModel
from my_seg_depth.trymulti.semantic_trans import networks5_ds
from my_seg_depth.trymulti.semantic_trans.networks5_ds import init_net, init_weights,GramMatrix
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

    def save_current_visuals(self):
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
            save_dir = './checkpoints/1_11'

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



        self.net_Dis0_en = networks5_ds.Discriminator2_seg().cuda()
        self.net_Dis0_en = init_net(self.net_Dis0_en)
        self.net_DIS = networks5_ds.Discriminator(repeat_num=5).cuda()
        self.net_DIS = init_net(self.net_DIS)

        self.net_G_1 = networks5_ds.G_1().cuda()
        self.net_G_1 = nn.DataParallel(self.net_G_1)
        self.net_G_1.load_state_dict(torch.load(
         './checkpoints/1_9_0_vt_t/iter_10000_net_G_1.pth'))


        print('1')

        self.net_G_2 = networks5_ds.General_net().cuda()

        self.net_G_2 = nn.DataParallel(self.net_G_2)
        self.net_G_2.load_state_dict(torch.load(
        './checkpoints/1_9_0_vt_t/iter_10000_net_G_2.pth'))
        print('2')

        self.net_Seg_de = networks5_ds.SEG(n_cls=28).cuda()
      #  self.net_Seg_de = init_net(self.net_Seg_de)
        self.net_Seg_de = nn.DataParallel(self.net_Seg_de)
        self.net_Seg_de.load_state_dict(torch.load(
            './checkpoints/1_9_0_vt_t/iter_10000_net_Seg_de.pth'))

        self.net_Dep_de = networks5_ds.DEP().cuda()
       # self.net_Dep_de = init_net(self.net_Dep_de)
        self.net_Dep_de = nn.DataParallel(self.net_Dep_de)
        self.net_Dep_de.load_state_dict(torch.load(
            './checkpoints/1_9_0_vt_t/iter_10000_net_Dep_de.pth'))
           # './checkpoints/1_3_0_vt_t/iter_70000_net_Dep_de.pth'))
        self.net_R_D = networks5_ds.R_dep().cuda()
        self.net_R_D = nn.DataParallel(self.net_R_D).cuda()
       # self.net_R_D = init_net(self.net_R_D)
        self.net_R_D.load_state_dict(torch.load(
            '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/checkpoints/1_9_0_vt_t/iter_10000_net_R_D.pth'
       #     '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/checkpoints/1_9_0_vt_t/iter_8000_net_R_D.pth'
        #    './checkpoints/1_1_4/iter_34000_net_R_D.pth'
       # './checkpoints/1_3_0_vt_t/iter_70000_net_R_D.pth'
        ))

        self.net_Dis_80 = networks5_ds.Discriminator(curr_dim=1, repeat_num=3).cuda()
        self.net_Dis_80 = init_net(self.net_Dis_80)

        self.net_Dis_160 = networks5_ds.Discriminator(curr_dim=1,repeat_num=4).cuda()
        self.net_Dis_160 = init_net(self.net_Dis_160)

        self.net_Dis_320 = networks5_ds.Discriminator(curr_dim=1,repeat_num=4).cuda()
        self.net_Dis_320 = init_net(self.net_Dis_320)

        self.optimizer_G_1 = torch.optim.Adam(self.net_G_1.parameters(),
                                            lr=opt.lr/2, betas=(opt.beta1, 0.999))
        self.optimizer_G_2 = torch.optim.Adam(self.net_G_2.parameters(),
                                              lr=opt.lr/2,betas=(opt.beta1,0.999))
        self.optimizer_Seg = torch.optim.Adam(self.net_Seg_de.parameters(),
                                              lr=opt.lr/2,betas=(opt.beta1,0.999))
        self.optimizer_Dep = torch.optim.Adam(self.net_Dep_de.parameters(),
                                            lr=opt.lr/2, betas=(opt.beta1, 0.999))
        self.optimizer_D0 = torch.optim.Adam(self.net_Dis0_en.parameters(),
                                            lr=opt.lr/2, betas=(opt.beta1, 0.999))
        self.optimizer_R_D = torch.optim.Adam(self.net_R_D.parameters(),
                                              lr=opt.lr/2, betas=(opt.beta1, 0.999)
                                              )

        self.optimizer_DIS = torch.optim.Adam(self.net_DIS.parameters(),
                                              lr=opt.lr / 4, betas=(opt.beta1, 0.999)
                                              )
        self.optimizer_Dis_80 = torch.optim.Adam(self.net_Dis_80.parameters(),
                                              lr=opt.lr / 4, betas=(opt.beta1, 0.999)
                                              )

        self.optimizer_Dis_160 = torch.optim.Adam(self.net_Dis_160.parameters(),
                                              lr=opt.lr / 3, betas=(opt.beta1, 0.999)
                                              )
        self.optimizer_Dis_320 = torch.optim.Adam(self.net_Dis_320.parameters(),
                                              lr=opt.lr / 3, betas=(opt.beta1, 0.999)
                                              )

        self.syn_imgpool = ImagePool(opt.pool_size)
        self.real_imgpool = ImagePool(opt.pool_size)

        self.criterionGAN = networks5_ds.GANLoss(use_lsgan =True)
        self.criterionSeg = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255)
        self.criterionDep =torch.nn.L1Loss()
        self.criterionStyle = torch.nn.MSELoss()
        self.criterionEdge = nn.BCELoss()
    def visual(self,train_or_test):
        if train_or_test=='train':
            self.loss_names = [  # 'G2_dis',
                'D_syn', 'dep_syn',  # 'DIS_real','DIS_syn',
                # 'seg_syn',
                'd320_real', 'd320_syn', 'd160_real', 'd160_syn',
                'seg_real',
                'D_real', 'dep_real',
                # 'style_0','style_1','style_2',
                'dep_ref',
                'DEP_real',
                'DEP_syn'  # 'adv_DIS_syn',
            ]
            self.visual_names = ['syn_img', 'real_img', 'syn_seg_l',  # 'real_seg_l',
                                 'syn_seg_pre', 'real_seg_pre', 'syn_dep_l', 'syn_dep_pre', 'real_dep_pre',
                                 'syn_dep_ref', 'real_dep_ref',
                                 ]
            self.model_names = ['G_1', 'G_2', 'Dis0_en',  # ,'Dis1_en',#'Dis2_en','Dis3_en',
                                'Dep_de',
                                'Seg_de',
                                'R_D'  # ,'DIS'
                                ]
        else:  # during test time, only load Gs
            self.loss_names = [  # 'G2_dis',
                'D_syn', 'dep_syn',  # 'DIS_real','DIS_syn',
                # 'seg_syn',
                'd320_real', 'd320_syn', 'd160_real', 'd160_syn',
               # 'seg_real',
                'D_real', 'dep_real',
                # 'style_0','style_1','style_2',
                'dep_ref',
                'DEP_real',
                'DEP_syn'  # 'adv_DIS_syn',
            ]
            self.visual_names = ['syn_img', 'real_img', 'syn_seg_l',  # 'real_seg_l',
                                 'syn_seg_pre', 'real_seg_pre', 'syn_dep_l', 'syn_dep_pre', 'real_dep_pre',
                                 'syn_dep_ref', 'real_dep_ref',
                                 ]
            self.model_names = ['G_1', 'G_2', 'Dep_de',
                                'Seg_de', 'R_D']

    def detach_list(self,list):
        for i in list:
            i = i.detach()
        return list
    def set_input(self,input,train_or_test):
        self.real_img = input['img_real'].cuda()
        self.syn_img = input['img_syn'].cuda()
        if train_or_test =='train':
            self.is_Train=True
            self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
            self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
            self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()
            self.real_img_paths = input['img_source_paths']
            self.syn_img_paths = input['img_target_paths']
            self.syn_dep_ls = input['depth_l_s'].float().cuda()
            self.syn_seg_le = input['seg_e_syn'].float().cuda()
            self.real_seg_le = input['seg_e_real'].float().cuda()
        else:
            self.is_Train=False
           # self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
            self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
            self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()
            self.real_img_paths = input['img_source_paths']
            self.syn_img_paths = input['img_target_paths']
            self.syn_dep_ls = input['depth_l_s'].float().cuda()
            self.syn_seg_le = input['seg_e_syn'].float().cuda()
            #self.real_seg_le = input['seg_e_real'].float().cuda()

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

    def backward_D(self):
        self.set_requires_grad([  # self.net_Dis3_en, self.net_Dis2_en,
            #self.net_Dis1_en,
            self.net_Dis0_en], True)
        self.optimizer_D0.zero_grad()
        #self.optimizer_D1.zero_grad()
    #    self.optimizer_D2.zero_grad()
     #   self.optimizer_D3.zero_grad()
        s_f = self.net_G_1(self.syn_img)
        syn_features1,syn_Features = self.net_G_2(s_f,'S')
        syn_features1  = syn_features1.detach()
        del syn_Features
        syn_features2=self.net_Seg_de(syn_features1)[1].detach()
        pre_s = self.net_Dis0_en(syn_features2)
        real_features1 = self.net_G_2(self.real_img,'R')[0].detach()

        #pre_s0 = self.net_Dis0_en(syn_features1[:, :256, :, :])
        #pre_s1 = self.net_Dis1_en(syn_features1[:, 256:512, :, :])
    #    pre_s2 = self.net_Dis2_en(syn_features1[:, 512:768, :, :])
     #   pre_s3 = self.net_Dis3_en(syn_features1[:, 768:1024, :, :])
        loss_D_syn0 = self.criterionGAN(pre_s, False)


  #      self.loss_D_syn2 = self.criterionGAN(pre_s2, True)
   #     self.loss_D_syn3 = self.criterionGAN(pre_s3, True)


        pre_r0 = self.net_Dis0_en(self.net_Seg_de(real_features1)[1].detach())

      #  pre_r2 = self.net_Dis2_en(real_features1[:, 512:768, :, :])
      #  pre_r3 = self.net_Dis3_en(real_features1[:, 768:1024, :, :])
        loss_D_real0 = self.criterionGAN(pre_r0, True)

        self.loss_D0 = loss_D_real0+ loss_D_syn0
        self.loss_D0.backward()
        print('dis update')
        self.optimizer_D0.step()

  #      self.loss_D_real2 = self.criterionGAN(pre_r2, False)
   #     self.loss_D_real3 = self.criterionGAN(pre_r3, False)
        self.loss_D_real = (loss_D_real0).detach()
        self.loss_D_syn = (loss_D_syn0
                           # + self.loss_D_syn2 + self.loss_D_syn3
                           ).detach()
        #+self.loss_D_real2+self.loss_D_real3).detach()

     #,self.loss_D_real2+self.loss_D_syn2,self.loss_D_real3+self.loss_D_syn3

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



  #  def cal_derp_loss_skyfree(self):
   #     sky_map=nn.softmax(self.syn_seg_pre[:,17,:,:],dim=1)

    def backward_Seg(self):

        seg_syn_pre,syn_features2 = self.net_Seg_de(self.syn_features1.detach())

        seg_real_pre = self.net_Seg_de(self.real_features1.detach())[0]
    #    self.loss_seg_syn = self.criterionSeg(seg_syn_pre,self.syn_seg_l)
        if self.is_Train:
            self.loss_seg_real = self.criterionSeg(seg_real_pre,self.real_seg_l)
        else:
            self.loss_seg_real=0
        self.syn_seg_pre = seg_syn_pre.detach()
        self.real_seg_pre = seg_real_pre.detach()

        #syn_features2 = self.net_Seg_de(self.syn_features1.detach())[1]
        pre_s = self.net_Dis0_en(syn_features2)

        # pre_s0 = self.net_Dis0_en(syn_features1[:, :256, :, :])
        # pre_s1 = self.net_Dis1_en(syn_features1[:, 256:512, :, :])
        #    pre_s2 = self.net_Dis2_en(syn_features1[:, 512:768, :, :])
        #   pre_s3 = self.net_Dis3_en(syn_features1[:, 768:1024, :, :])
        loss_D_syn0 = self.criterionGAN(pre_s, True)

        #      self.loss_D_syn2 = self.criterionGAN(pre_s2, True)
        #     self.loss_D_syn3 = self.criterionGAN(pre_s3, True)



        return self.loss_seg_real+3*loss_D_syn0#+self.loss_seg_syn#+0.05*self.loss_adv_DIS_syn




    def real_dep_loss(self,seg_p, seg_l, dep_p, dep_l):
        seg_l =seg_l.float().cuda()
        seg_p = seg_p.detach()
        new_seg_p = seg_p.max(dim=1)[1].float().cuda()

        print(torch.max(new_seg_p),torch.max(seg_l))
        nn = torch.zeros(new_seg_p.shape).float().cuda()
        nn[new_seg_p == seg_l] = 1
        new_dep_p = (nn * dep_p).float().cuda()
        dep_l = nn * dep_l
        print(torch.max(dep_l),torch.max(new_dep_p))
        loss = self.criterionDep(new_dep_p,dep_l)

        return loss

    def backward_DISDEP(self):
        self.set_requires_grad([self.net_G_1, self.net_G_2,
                                self.net_Seg_de, self.net_Dep_de,self.net_R_D], False)
        self.set_requires_grad([self.net_DIS,self.net_Dis_160,self.net_Dis_320],True)
        self.optimizer_DIS.zero_grad()
        if self.is_Train:
            D_real = self.net_DIS(torch.cat([self.real_dep_ref,self.real_seg_l.detach().unsqueeze(1).float()],1))
            self.loss_DEP_real = self.criterionGAN(D_real, True).detach()
        else:
            D_real = 1

        D_fake = self.net_DIS(torch.cat([self.syn_dep_ref,self.syn_seg_l.detach().unsqueeze(1).float()],1))


        self.loss_DEP_syn = self.criterionGAN(D_fake,False).detach()

        (self.criterionGAN(D_real, True) + self.criterionGAN(D_fake, False)).backward()

        self.optimizer_DIS.step()
        print('DIS update')
        self.optimizer_Dis_160.zero_grad()
        D_real_160 = self.net_Dis_160(self.real_dep_160)
        D_syn_160 = self.net_Dis_160(self.syn_dep_160)
        self.loss_d160_real = self.criterionGAN(D_real_160,True).detach()
        self.loss_d160_syn = self.criterionGAN(D_syn_160,True).detach()
        (self.criterionGAN(D_real_160,True)+self.criterionGAN(D_syn_160,False)).backward()
        _ = torch.nn.utils.clip_grad_norm_(self.net_Dis_160.parameters(), max_norm=1.0)
        self.optimizer_Dis_160.step()
        self.optimizer_Dis_320.zero_grad()
        D_real_320 = self.net_Dis_320(self.real_dep_320)
        D_syn_320 = self.net_Dis_320(self.syn_dep_320)
        self.loss_d320_real = self.criterionGAN(D_real_320, True).detach()
        self.loss_d320_syn = self.criterionGAN(D_syn_320, True).detach()
        (self.criterionGAN(D_real_320, True) + self.criterionGAN(D_syn_320, False)).backward()
        _ = torch.nn.utils.clip_grad_norm_(self.net_Dis_320.parameters(),max_norm=1.0)
        self.optimizer_Dis_320.step()
        self.set_requires_grad([self.net_DIS, self.net_Dis_160, self.net_Dis_320], False)


    def test_return(self):
        return self.real_img,self.real_dep_ref
    def backward_R_D(self,train_or_test):
        self.optimizer_R_D.zero_grad()
        A = True
        if A:
            r_Seds, r_Segs, r_Deps = self.net_R_D(self.detach_list(self.real_Features),self.real_features1.detach())

            se_loss_real = 0
            seg_loss_real = 0
            if self.is_Train:

                for sed in r_Seds:
                    se_loss_real =se_loss_real+ se_loss_real+self.criterionEdge(sed[:,0,:,:],self.real_seg_le)
                for seg in r_Segs:
                    seg_loss_real =se_loss_real+ self.criterionSeg(seg,self.real_seg_l)
                print(r_Segs[2].shape)


            real_dep_160 = nn.UpsamplingBilinear2d(scale_factor=0.25)(r_Seds[0][:,1,:,:].unsqueeze(1))
            real_dep_320 = nn.UpsamplingBilinear2d(scale_factor=0.5)(r_Seds[1][:, 1, :, :].unsqueeze(1))
            D_real_160=self.net_Dis_160(real_dep_160)
            D_real_320 = self.net_Dis_320(real_dep_320)
            real_dep_ref = r_Seds[2][:,1,:,:].unsqueeze(1)


            D_real_loss =se_loss_real+seg_loss_real+self.criterionGAN(D_real_160,False) +self.criterionGAN(D_real_320,False)#+ seg_loss_real#+se_loss_real


            self.real_dep_ref = real_dep_ref.detach()
            self.real_dep_160 = real_dep_160.detach()
            self.real_dep_320 = real_dep_320.detach()



            if train_or_test=='train':
                D_real_loss.backward()
                self.optimizer_R_D.step()

        self.optimizer_R_D.zero_grad()

        s_Seds, s_Segs, s_Deps = self.net_R_D(self.detach_list(self.syn_Features),self.syn_features1.detach())


        s_se_loss = 0
        s_seg_loss = 0
        dep_loss=0
        B = True
        if B:
            for sed in s_Seds:
                s_se_loss =s_se_loss+self.criterionEdge(sed[:,0,:,:],self.syn_seg_le)
                dep_loss = dep_loss+self.criterionDep(sed[:,1,:,:],self.syn_dep_l)
            for seg in s_Segs:
                s_seg_loss =s_seg_loss+ self.criterionSeg(seg,self.syn_seg_l)
            for s_Dep in s_Deps:
                dep_loss = dep_loss+self.criterionDep(s_Dep,self.syn_dep_ls)




        syn_dep_160 = nn.UpsamplingBilinear2d(scale_factor=0.25)(s_Seds[0][:,1,:,:].unsqueeze(1))
        syn_dep_320 = nn.UpsamplingBilinear2d(scale_factor=0.5)(s_Seds[1][:, 1, :, :].unsqueeze(1))
       # D_real_160=self.net_Dis_160(real_dep_160)
       # D_real_320 = self.net_Dis_320(real_dep_320)
        syn_dep_ref = s_Seds[2][:,1,:,:].unsqueeze(1)
        loss_dep_ref = self.criterionDep(s_Seds[2][:,1,:,:],self.syn_dep_l)
        D_syn_loss = 5* loss_dep_ref+s_se_loss+s_seg_loss
        if train_or_test == 'train':
            D_syn_loss.backward()
            self.optimizer_R_D.step()
        self.loss_dep_ref = loss_dep_ref.detach()
       # print('loss_dep',self.loss_dep_ref)


        self.syn_dep_ref = syn_dep_ref.detach()
        self.syn_dep_160 = syn_dep_160.detach()
        self.syn_dep_320 = syn_dep_320.detach()

        

       # dis_512_real = self.net_Dis_512(syn_style_F[0])

        return D_syn_loss#+#D_real_loss+D_syn_loss#+D_syn_loss#+10**-30*(loss_style_0+loss_style_1+loss_style_2)






    def backward_Dep(self):
        syn_features2 = self.syn_features1.detach()
        dep_syn_pre = self.net_Dep_de(syn_features2.detach())
        self.loss_dep_syn =  self.criterionDep(dep_syn_pre,self.syn_dep_l)
        self.syn_dep_pre = dep_syn_pre.detach()

        return self.loss_dep_syn

    def backward_G_1(self):
        self.set_requires_grad([self.net_Seg_de,self.net_G_2,self.net_Dep_de], False)
        ss = self.net_G_1(self.syn_img)
        self.syn_features1,self.syn_Features = self.net_G_2(ss,'S')
        seg_syn_pre, syn_features2 = self.net_Seg_de(self.syn_features1)
      #  self.syn_features1_ = self.net_G_1(self.syn_img  )
        print('pre_dep_f',syn_features2.shape)



        pre_s = self.net_Dis0_en(syn_features2)

        # pre_s0 = self.net_Dis0_en(syn_features1[:, :256, :, :])
        # pre_s1 = self.net_Dis1_en(syn_features1[:, 256:512, :, :])
        #    pre_s2 = self.net_Dis2_en(syn_features1[:, 512:768, :, :])
        #   pre_s3 = self.net_Dis3_en(syn_features1[:, 768:1024, :, :])
        loss_D_syn0 = self.criterionGAN(pre_s, True)

        #loss_dep = self.criterionDep(dep_syn_pre, self.syn_dep_l)

        loss_seg_syn = self.criterionSeg(seg_syn_pre, self.syn_seg_l)
        #seg_syn_pre, seg_syn_feaetures3 = self.net_Seg_de(syn_f2, syn_inf)

        self.loss_G_1 =  loss_seg_syn+3*loss_D_syn0


        return self.loss_G_1

    def backward_G_2(self):
        self.set_requires_grad([self.net_Seg_de, self.net_G_1, self.net_Dep_de], False)
        self.set_requires_grad([#self.net_Dis3_en,self.net_Dis2_en,
            #                    self.net_Dis1_en,
            self.net_Dis0_en], False)
        ss = self.net_G_1(self.syn_img)
        self.real_features1,self.real_Features = self.net_G_2(self.real_img,'R')


        seg_real_pre,seg_real_f2 = self.net_Seg_de(self.real_features1)
        if self.is_Train:
            loss_seg = self.criterionSeg(seg_real_pre, self.real_seg_l)
            dep_real_pre = self.net_Dep_de(seg_real_f2.detach())
            del seg_real_f2
            self.real_dep_pre = dep_real_pre.detach()
           # self.loss_dep_syn = self.criterionDep(self.syn_dep_pre, self.syn_dep_l)
            loss_dep_real = self.real_dep_loss(seg_p=seg_real_pre,
                                                    seg_l=self.syn_seg_l,
                                                    dep_p=dep_real_pre, dep_l=self.syn_dep_l)

            # seg_syn_pre, seg_syn_feaetures3 = self.net_Seg_de(syn_f2, syn_inf)

            self.loss_dep_real = loss_dep_real.detach()
            self.loss_G_2 = loss_seg
        else:
            self.loss_G_2=0

        self.syn_features1, self.syn_Features = self.net_G_2(ss.detach(),'S')
        seg_syn_pre, syn_features2 = self.net_Seg_de(self.syn_features1)
        #  self.syn_features1_ = self.net_G_1(self.syn_img  )

        #print('pre_dep_f', syn_features2.shape)

        pre_s = self.net_Dis0_en(syn_features2)

        loss_D_syn0 = self.criterionGAN(pre_s, True)

       # loss_dep = self.criterionDep(dep_syn_pre, self.syn_dep_l)

        loss_seg_syn = self.criterionSeg(seg_syn_pre, self.syn_seg_l)
        loss_G_1 = loss_seg_syn + 5 * loss_D_syn0

        return self.loss_G_2+loss_G_1

    def optimize_parameters(self,train_or_test):



        self.set_requires_grad([self.net_G_1, self.net_G_2,
                                self.net_Seg_de], True)

        self.set_requires_grad([self.net_Seg_de, self.net_G_2,
                                #self.net_Dis3_en, self.net_Dis2_en,
                               # self.net_Dis1_en,
                                self.net_Dis0_en], False)
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

            print('seg update')
            self.loss_seg.backward()
            self.optimizer_Seg.step()

        self.optimizer_Dep.zero_grad()
        self.loss_dep= self.backward_Dep()


        if (train_or_test == 'train'):

            print('-----dep update')
            self.loss_dep.backward()
            self.optimizer_Dep.step()

        self.set_requires_grad([self.net_G_1, self.net_G_2,
                                self.net_Seg_de, self.net_Dep_de], False)
        self.set_requires_grad(self.net_R_D,True)
        loss_R_D = self.backward_R_D(train_or_test)
        if (train_or_test=='train'):
#            loss_R_D.backward()
 #           self.optimizer_R_D.step()
            print('R_D update')


        if (train_or_test == 'train'):
            self.set_requires_grad([#self.net_Dis3_en, self.net_Dis2_en,
                #                    self.net_Dis1_en,
                self.net_Dis0_en], True)
            self.set_requires_grad([self.net_G_1, self.net_G_2,
                                    self.net_Seg_de, self.net_Dep_de], False)
            self.set_requires_grad(self.net_R_D, False)


         #   self.optimizer_D3.zero_grad()
            self.backward_D()
            self.backward_DISDEP()








