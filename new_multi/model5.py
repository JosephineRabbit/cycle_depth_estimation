import torch
import itertools
from util.image_pool import ImagePool
#from .base_model import BaseModel
from new_multi import networks5_ds
from new_multi.networks5_ds import init_net, init_weights,get_masks
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
        self.dep_ref_name = []
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
                #print(name,visual_data.shape)
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
    def get_eval_visuals(self):
        visual_ret = OrderedDict()
        for name in self.dep_ref_name:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_data=value[-1].data
                else:
                  visual_data=value.data
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
            save_dir = './checkpoints/1_21'

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

    def initialize(self,opt):
        BaseModel.initialize(self,opt)


        self.net_FD1 = networks5_ds._Discriminator(input_nc=512).cuda()
        self.net_FD2 = networks5_ds._Discriminator(input_nc=256).cuda()
        self.net_FD3 = networks5_ds._Discriminator(input_nc=128).cuda()
        self.net_FD1 = init_net(self.net_FD1)
        self.net_FD2 = init_net(self.net_FD2)
        self.net_FD3 = init_net(self.net_FD3)
        #self.net_FD0 = networks5_ds._Discriminator(input_nc=64).cuda()

        self.net_G_1 = networks5_ds.G_1().cuda()
        self.net_G_1 = nn.DataParallel(self.net_G_1)
        self.net_G_1.load_state_dict(torch.load(
         '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/checkpoints/1_14/iter_8000_net_G_1.pth'))


        print('1')

        self.net_G_2 = networks5_ds.General_net().cuda()

        self.net_G_2 = nn.DataParallel(self.net_G_2)
        self.net_G_2.load_state_dict(torch.load(
        '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/checkpoints/1_14/iter_8000_net_G_2.pth'))
        print('2')

      #   self.net_Seg_de = networks5_ds.SEG(n_cls=28).cuda()
      # #  self.net_Seg_de = init_net(self.net_Seg_de)
      #   self.net_Seg_de = nn.DataParallel(self.net_Seg_de)
      #   self.net_Seg_de.load_state_dict(torch.load(
      #       './checkpoints/1_14/iter_8000_net_Seg_de.pth'))
      #
      #   self.net_Dep_de = networks5_ds.DEP().cuda()
      #  # self.net_Dep_de = init_net(self.net_Dep_de)
      #   self.net_Dep_de = nn.DataParallel(self.net_Dep_de)
      #   self.net_Dep_de.load_state_dict(torch.load(
      #       './checkpoints/1_14/iter_8000_net_Dep_de.pth'))
           # './checkpoints/1_3_0_vt_t/iter_70000_net_Dep_de.pth'))
        self.net_R_D = networks5_ds.R_dep().cuda()
       # self.net_R_D = nn.DataParallel(self.net_R_D).cuda()
        self.net_R_D = init_net(self.net_R_D)
      #  self.net_R_D.load_state_dict(torch.load(
       #     '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/checkpoints/1_14/iter_8000_net_R_D.pth'
       #     '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/my_seg_depth/trymulti/semantic_trans/checkpoints/1_14/iter_8000_net_R_D.pth'
        #    './checkpoints/1_1_4/iter_34000_net_R_D.pth'
       # './checkpoints/1_3_0_vt_t/iter_70000_net_R_D.pth'
        #))



        self.optimizer_G_1 = torch.optim.Adam(self.net_G_1.parameters(),
                                            lr=opt.lr/5, betas=(opt.beta1, 0.999))
        self.optimizer_G_2 = torch.optim.Adam(self.net_G_2.parameters(),
                                              lr=opt.lr/3,betas=(opt.beta1,0.999))
        # self.optimizer_Seg = torch.optim.Adam(self.net_Seg_de.parameters(),
        #                                       lr=opt.lr/2,betas=(opt.beta1,0.999))
        # self.optimizer_Dep = torch.optim.Adam(self.net_Dep_de.parameters(),
        #                                     lr=opt.lr/2, betas=(opt.beta1, 0.999))

        self.optimizer_R_D = torch.optim.Adam(self.net_R_D.parameters(),
                                              lr=opt.lr/2, betas=(opt.beta1, 0.999)
                                              )


        # self.optimizer_FD0 = torch.optim.Adam(self.net_FD0.parameters(),
        #                                       lr=opt.lr/4, betas=(opt.beta1, 0.999)
        #                                       )
        self.optimizer_FD1 = torch.optim.Adam(self.net_FD1.parameters(),
                                              lr=opt.lr/4, betas=(opt.beta1, 0.999)
                                              )
        self.optimizer_FD2 = torch.optim.Adam(self.net_FD2.parameters(),
                                              lr=opt.lr/4, betas=(opt.beta1, 0.999)
                                              )
        self.optimizer_FD3 = torch.optim.Adam(self.net_FD3.parameters(),
                                              lr=opt.lr/4, betas=(opt.beta1, 0.999)
                                              )

        self.syn_imgpool = ImagePool(opt.pool_size)
        self.real_imgpool = ImagePool(opt.pool_size)

        self.criterionGAN = networks5_ds.GANLoss(use_lsgan =True)
        self.criterionSeg = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255)
        self.criterionDep =torch.nn.L1Loss()
        self.criterionStyle = torch.nn.MSELoss()
     #   self.criterionEdge = nn.BCELoss()
        self.criterionDep_bce =  networks5_ds.BCEDepLoss()

    def visual(self,train_or_test):
        if train_or_test=='train':
            self.loss_names = [

            ]
            self.visual_names = ['syn_img', 'real_img', 'syn_seg_l',  # 'real_seg_l',
                                 #'syn_seg_pre', 'real_seg_pre',
                                 'syn_dep_l',
                                 #'syn_dep_pre', 'real_dep_pre',
                                 'syn_dep_ref', 'real_dep_ref',
                             ]
            self.dep_ref_name = ['real_dep_ref']
            self.model_names = ['G_1', 'G_2',# 'Dis0_en',  # ,'Dis1_en',#'Dis2_en','Dis3_en',
                                #'Dep_de',
                                #'Seg_de',
                                'R_D'  # ,'DIS'
                                ]
        else:  # during test time, only load Gs
            self.loss_names = [  # 'G2_dis',

            ]
            self.visual_names = ['syn_img', 'real_img', 'syn_seg_l',  # 'real_seg_l',
                                 #'syn_seg_pre', 'real_seg_pre',
                                 'syn_dep_l',
                                 #'syn_dep_pre', 'real_dep_pre',
                                 'syn_dep_ref', 'real_dep_ref',
                                 ]
            self.dep_ref_name = ['real_dep_ref']
            self.model_names = ['G_1', 'G_2', #'Dep_de',
                                #'Seg_de',
                                'R_D']

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
            self.f_name = input['f_name']
            self.l_name = input['l_name']
            self.is_Train=False
           # self.real_seg_l = input['seg_l_real'].squeeze(1).cuda()
            self.syn_seg_l = input['seg_l_syn'].squeeze(1).cuda()
            self.syn_dep_l = input['dep_l_syn'].squeeze(1).cuda()
            self.real_img_paths = input['img_source_paths']
            self.syn_img_paths = input['img_target_paths']
            self.syn_dep_ls = input['depth_l_s'].float().cuda()
            self.syn_seg_le = input['seg_e_syn'].float().cuda()
            #self.real_seg_le = input['seg_e_real'].float().cuda()

    def return_name(self):
        return self.f_name,self.l_name
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
            self.net_FD0], True)
        self.optimizer_FD0.zero_grad()
        #self.optimizer_D1.zero_grad()
    #    self.optimizer_D2.zero_grad()
     #   self.optimizer_D3.zero_grad()
        s_f = self.net_G_1(self.syn_img)
        #syn_features1,syn_Features = self.net_G_2(s_f,'S')

        #del syn_Features


        pre_s= self.net_FD0(s_f)

        loss_D_syn0 = self.criterionGAN(pre_s, False)

        pre_r0 = self.net_FD0(self.real_Features[0])

      #  pre_r2 = self.net_Dis2_en(real_features1[:, 512:768, :, :])
      #  pre_r3 = self.net_Dis3_en(real_features1[:, 768:1024, :, :])
        loss_D_real0 = self.criterionGAN(pre_r0, True)

        self.loss_D0 = loss_D_real0+ loss_D_syn0
        self.loss_D0.backward()
        print('DF0 update')
        self.optimizer_FD0.step()

  #      self.loss_D_real2 = self.criterionGAN(pre_r2, False)
   #     self.loss_D_real3 = self.criterionGAN(pre_r3, False)
        self.loss_D_real = (loss_D_real0).detach()
        self.loss_D_syn = (loss_D_syn0).detach()
        #+self.loss_D_real2+self.loss_D_real3).detach()






    def backward_DISDEP(self):

        # self.set_requires_grad([self.net_FD1,self.net_FD2,self.net_FD3],True)
        # feats, seg, (dep_4, dep_o) = self.net_R_D(self.real_Features, self.real_features1)
        # real_feats = self.detach_list(feats)
        # del feats,seg,(dep_4,dep_o)
        # s_feats, seg, (dep_4, dep_o) = self.net_R_D(self.syn_Features, self.syn_features1)
        # syn_feats = self.detach_list(s_feats)
        # del s_feats, seg, (dep_4, dep_o)
        # self.optimizer_R_D.zero_grad()
        # self.optimizer_R_D.step()
        #self.optimizer_.zero_grad()
        self.optimizer_FD1.zero_grad()

        D_real = self.net_FD1(self.real_feats[0].detach())
        D_fake = self.net_FD1(self.syn_feats[0].detach())

        loss = self.criterionGAN(D_real, True) + self.criterionGAN(D_fake, False)
        loss.backward()

        self.optimizer_FD1.step()
        self.loss_DEP_syn = self.criterionGAN(D_fake, False).detach()
        self.loss_DEP_real = self.criterionGAN(D_real, True).detach()
        print('FD1 update')
        del D_real,D_fake

        self.optimizer_FD2.zero_grad()
        D_real = self.net_FD2(self.real_feats[1].detach())
        self.loss_DEP_real += self.criterionGAN(D_real, True).detach()


        D_fake = self.net_FD2(self.syn_feats[1].detach())

        self.loss_DEP_syn += self.criterionGAN(D_fake, False).detach()

        (self.criterionGAN(D_real, True) + self.criterionGAN(D_fake, False)).backward()

        self.optimizer_FD2.step()
        self.loss_DEP_syn += self.criterionGAN(D_fake, False).detach()
        self.loss_DEP_real += self.criterionGAN(D_real, True).detach()
        del D_fake,D_real

        print('FD2 update')

        self.optimizer_FD3.zero_grad()
        D_real = self.net_FD3(self.real_feats[2].detach())
        self.loss_DEP_real = self.criterionGAN(D_real, True).detach()

        D_fake = self.net_FD3(self.syn_feats[2].detach())

        self.loss_DEP_syn = self.criterionGAN(D_fake, False).detach()

        (self.criterionGAN(D_real, True) + self.criterionGAN(D_fake, False)).backward()

        self.optimizer_FD3.step()
        self.loss_DEP_syn += self.criterionGAN(D_fake, False).detach()
        self.loss_DEP_real += self.criterionGAN(D_real, True).detach()
        del D_fake, D_real

        self.set_requires_grad([self.net_FD1,self.net_FD2,self.net_FD3], False)


    def test_return(self):
        return self.real_img,self.real_dep_ref
    def backward_R_D(self,train_or_test):
        self.optimizer_R_D.zero_grad()
        A = True
        if A:
            feats,seg,(dep_4,dep_o) = self.net_R_D(self.real_Features,self.real_features1)
            self.real_feats = self.detach_list(feats)

            seg_loss_real = 0
            if self.is_Train:

             #   for sed in r_Seds:
              #      se_loss_real =se_loss_real+ se_loss_real+self.criterionEdge(sed[:,0,:,:],self.real_seg_le)
                seg_loss_real = self.criterionSeg(seg,self.real_seg_l)

            pred1 = self.net_FD1(feats[0])
            pred2 = self.net_FD2(feats[1])
            pred3 = self.net_FD3(feats[2])



            D_real_loss =seg_loss_real+0.2*self.criterionGAN(pred1,False) \
                         +0.2*self.criterionGAN(pred2,False)+0.2*self.criterionGAN(pred3,False)#+ seg_loss_real#+se_loss_real


            self.real_dep_ref = dep_o.squeeze(1).detach()




            if train_or_test=='train':
                D_real_loss.backward()
                self.optimizer_R_D.step()
            self.real_feats = self.detach_list(feats)

        self.optimizer_R_D.zero_grad()
        del feats,seg,(dep_4,dep_o)

        feats, seg, (dep_4, dep_o) = self.net_R_D(self.syn_Features,self.syn_features1)



        s_se_loss = 0
        s_seg_loss = 0
        dep_loss=0

        #o_m, z_m = get_masks(self.syn_dep_l.clone())

        B = True
        if B:
            sky_m= self.syn_seg_l.clone()
            sky_m[sky_m!=17]=1
            sky_m[sky_m==17]=0
            print(sky_m.shape,'sky_m')
            oms, zms = get_masks(torch.cat([sky_m.unsqueeze(1),sky_m.unsqueeze(1),
                                            sky_m.unsqueeze(1),sky_m.unsqueeze(1)],1).float()*self.syn_dep_ls.clone())


            dep_loss = self.criterionDep(dep_o,sky_m.float()*self.syn_dep_l)
            s_seg_loss = self.criterionSeg(seg,self.syn_seg_l)
            for s_Dep in dep_4:
                dep_loss = dep_loss+self.criterionDep_bce(sky_m.unsqueeze(1).float()*s_Dep,torch.cat([sky_m.unsqueeze(1),sky_m.unsqueeze(1),
                                            sky_m.unsqueeze(1),sky_m.unsqueeze(1)],1).float()*self.syn_dep_ls.clone(),oms,zms)



       # D_real_160=self.net_Dis_160(real_dep_160)
       # D_real_320 = self.net_Dis_320(real_dep_320)
        syn_dep_ref = dep_o.squeeze(1)

        D_syn_loss = dep_loss+s_se_loss+s_seg_loss#+dep_loss
        if train_or_test == 'train':
            D_syn_loss.backward()
            self.optimizer_R_D.step()
        self.loss_dep_ref = dep_loss.detach()
       # print('loss_dep',self.loss_dep_ref)


        self.syn_dep_ref = syn_dep_ref.detach()
        self.syn_feats = self.detach_list(feats)

        return D_syn_loss#+#D_real_loss+D_syn_loss#+D_syn_loss#+10**-30*(loss_style_0+loss_style_1+loss_style_2)




    def backward_G_1(self):
        self.set_requires_grad([self.net_R_D,self.net_G_2], False)
        self.set_requires_grad([self.net_G_1],True)
        ss = self.net_G_1(self.syn_img)
        syn_features1, syn_Features = self.net_G_2(ss, 'S')

        s_feats,s_seg,(s_dep_4,s_dep_o) = self.net_R_D(syn_Features,syn_features1)



        loss_dep = self.criterionDep(s_dep_o, self.syn_dep_l)

        loss_seg_syn = self.criterionSeg(s_seg, self.syn_seg_l)


        loss_G_1 =  loss_seg_syn+loss_dep



        return loss_G_1

    def backward_G_2(self):
        self.set_requires_grad([self.net_R_D, self.net_G_1], False)
        self.set_requires_grad([#self.net_Dis3_en,self.net_Dis2_en,
            #                    self.net_Dis1_en,
            #self.net_FD0,
            self.net_FD1,self.net_FD2], False)

        ss = self.net_G_1(self.syn_img)
        syn_features1, syn_Features = self.net_G_2(ss.detach(), 'S')

        feats, seg, (dep_4, dep_o) = self.net_R_D(syn_Features,syn_features1)
        self.syn_feats = self.detach_list(feats)

        s_seg_loss = 0
        dep_loss = 0

        B = True
        if B:
            sky_m = self.syn_seg_l.clone()
            sky_m[sky_m != 17] = 1
            sky_m[sky_m == 17] = 0
            print(sky_m.shape, 'sky_m')

            dep_loss = self.criterionDep(dep_o, sky_m.float() * self.syn_dep_l)
            s_seg_loss = self.criterionSeg(seg, self.syn_seg_l)


        D_syn_loss = dep_loss + s_seg_loss
        self.syn_features1 = syn_features1.detach()
        self.syn_Features = self.detach_list(syn_Features)
        del syn_features1,syn_Features,feats


        real_features1,real_Features = self.net_G_2(self.real_img,'R')

        feats, seg, (dep_4, dep_o) = self.net_R_D(real_Features, real_features1)

        self.real_features1 = real_features1.detach()
        self.real_Features = self.detach_list(real_Features)
        del (dep_4,dep_o),feats

        seg_loss_real = 0
        if self.is_Train:
            #   for sed in r_Seds:
            #      se_loss_real =se_loss_real+ se_loss_real+self.criterionEdge(sed[:,0,:,:],self.real_seg_le)
            seg_loss_real = self.criterionSeg(seg, self.real_seg_l)



        D_real_loss = seg_loss_real



        return D_syn_loss+2*D_real_loss

    def optimize_parameters(self,train_or_test):

        self.set_requires_grad(self.net_G_2, True)
        self.optimizer_G_2.zero_grad()

        self.loss_G2 = self.backward_G_2()

        if (train_or_test == 'train'):
            print('g2_update')
            self.loss_G2.backward()
            self.optimizer_G_2.step()


        self.set_requires_grad([self.net_G_1, #self.net_G_2,
                                #self.net_Seg_de
                                ], True)

        self.set_requires_grad([#self.net_Seg_de,
                                self.net_G_2,
                                #self.net_Dis3_en, self.net_Dis2_en,
                               # self.net_Dis1_en,

                                ], False)
        self.optimizer_G_1.zero_grad()
        self.loss_G1 =self.backward_G_1()

        if (train_or_test == 'train'):
            print('g1_update')
            self.loss_G1.backward()
            self.optimizer_G_1.step()




       # self.set_requires_grad([self.net_Seg_de, self.net_Dep_de],True)
        self.set_requires_grad([self.net_G_1, self.net_G_2],False)#,self.net_DIS], False)




        self.set_requires_grad([self.net_G_1, self.net_G_2], False)
        self.set_requires_grad(self.net_R_D,True)
        loss_R_D = self.backward_R_D(train_or_test)
        if (train_or_test=='train'):

            print('R_D update')


        if (train_or_test == 'train'):

            self.set_requires_grad([self.net_G_1, self.net_G_2,
                                    self.net_R_D], False)
            self.set_requires_grad([
                self.net_FD1,self.net_FD2,self.net_FD3], True)

            #self.backward_D()
            self.backward_DISDEP()








