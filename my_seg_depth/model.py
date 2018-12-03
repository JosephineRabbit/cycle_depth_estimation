import torch
import itertools
from util.image_pool import ImagePool
#from .base_model import BaseModel
from . import networks
#from .encoder_decoder import _UNetEncoder,_UNetDecoder
import torch.nn as nn
from util.util import scale_pyramid
import os
import util.util as util
from collections import OrderedDict
import functools
import torch.nn.functional as F

class BaseModel():

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self,opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_Train  = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0,1])) \
            if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.chaeckjpoints_dir,opt.name)
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
                segname_syn=['lab_s','lab_s_pre','lab_A','segAreal','segAfake']
                segname_real=['lab_t','lab_t_pre','lab_B','segBreal','segBfake']
                if (name in segname_syn):
                    visual_ret[name]=util.label2im(visual_data,syn_or_real='syn')
                elif (name in segname_real):
                    visual_ret[name]=util.label2im(visual_data,syn_or_real='real')
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

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
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


class Seg_Depth(BaseModel):

    def name(self):
        return 'Seg_Depth_Model'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)

        self.loss_names=['G1_dis','G1_seg','D_G1',
                         'G2_dis','G2_seg','D_G2',
                         'Fnet_seg_s','Fnet_seg_r','Fnet_dep'
                         'DE_seg_s','DE_seg_r','DE_dep']

        self.visual_names = ['syn_img','real_img','syn_seg_lable','real_seg_label'
                             'syn_seg_pre','real_seg_pre'
                             'syn_dep_label','syn_dep_pre']

        if self.is_Train:
            self.model_names = ['G_1', 'G_2', 'Dis_en',
                                'Feature','Seg_de','Dep_de'
                                ]
        else:  # during test time, only load Gs
            self.model_names = ['G_1', 'G_2'
                                'Feature','Seg_de','Dep_de']

        self.net_G_1 = networks.define_G(opt.input_nc, opt.output_nc,  opt.netG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.net_G_2 = networks.define_G(opt.input_nc, opt.output_nc,  opt.netG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.net_Dis_en = networks.define_D()

        self.net_Feature = networks.Feature_net(input_nc=128,mid_nc =1024).cuda()
        self.net_Feature = nn.DataParallel(self.net_Feature)

        self.net_Seg_de = networks.SEG().cuda()
        self.net_Seg_de = nn.DataParallel(self.net_Seg_de)

        self.net_Dep_de = networks.DEP().cuda()
        self.net_Dep_de = nn.DataParallel(self.net_Dep_de)

        if self.is_Train:
            self.syn_imgpool - ImagePool(opt.poo_size)
            self.real_imgpool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(use_lsgan =not opt.no_lsgan).to(self.device())
            self.criterionSeg = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255).cuda()
            self.criterionDep = torch.nn.L1loss()


        Seg_loss = self.criterionSeg(output[-1], gt.squeeze(1))





