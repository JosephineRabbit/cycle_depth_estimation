import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import seg_network
import torch.nn as nn
import numpy as np

class SegModel(BaseModel):
    def name(self):
        return 'T2Net model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['loss_lab_s', 'loss_lab_t', 'acc_real', 'acc_syn']
        self.visual_names = ['img_s', 'img_t', 'lab_s_pre', 'lab_s', 'img_s2t', 'lab_t_pre', 'lab_t']

        self.model_names = ['img2task']

        # define the transform network
        self.net_s2t = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.net_s2t.eval()
        # define the task network
        self.net_img2task = seg_network.define_G(3, 20, opt.ngf, 4, opt.norm,
                                                 'PReLU', 'UNet', opt.init_type, 0,
                                             False,self.gpu_ids,0.1)

        if self.isTrain:
            self.fake_img_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.l1loss = torch.nn.L1Loss()
            self.nonlinearity = torch.nn.ReLU()
            # initialize optimizers
            self.optimizer_T2Net = torch.optim.Adam(self.net_img2task.parameters(),lr= 1e-3,betas=(0.95, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_T2Net)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source']
        self.img_target = input['img_target']
        self.lab_source = input['lab_source']
        self.lab_target = input['lab_target']

    def forward(self):
        self.img_s = Variable(self.img_source).cuda()
        self.img_t = Variable(self.img_target).cuda()
        self.lab_s = Variable(self.lab_source).cuda()
        self.lab_t = Variable(self.lab_target).cuda()

    def backward_translated2depth(self):
        # task network
        with torch.no_grad():
            self.img_s2t=self.net_s2t(self.img_s)
        fake = self.net_img2task.forward(self.img_s2t, syn_or_real='syn')
        self.lab_f_s = fake[0]  # feature
        self.lab_s_pre = fake[1]  # depth_prediction
        # task loss
        criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=255).cuda()
        task_loss = criterion(self.lab_s_pre, self.lab_s.squeeze(1))
        validmap = torch.where(self.lab_s == 255, torch.full_like(self.lab_s, 0),
                               torch.full_like(self.lab_s, 1)).data.cpu().numpy()[0, 0, :, :]
        pre = torch.argmax(self.lab_s_pre, dim=1).data.cpu().numpy()[0, :, :]
        gt = self.lab_s.data.cpu().numpy()[0, 0, :, :]
        acc_syn = np.sum((pre == gt) * validmap / np.sum(validmap))
        return task_loss, acc_syn

    def backward_real2depth(self):

        # image2depth
        fake = self.net_img2task.forward(self.img_t, syn_or_real='real')  # here use the original rgb
        # Gan depth
        self.lab_f_t = fake[0]
        self.lab_t_pre = fake[1]

        # img_real = task.scale_pyramid(self.img_t, size - 1)
        # self.loss_lab_smooth = task.get_smooth_weight(self.lab_t_g, img_real, size-1) * self.opt.lambda_smooth
        # total_loss =  self.loss_lab_smooth
        criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=255).cuda()
        task_loss = criterion(self.lab_t_pre, self.lab_t.squeeze(1))
        validmap = torch.where(self.lab_t == 255, torch.full_like(self.lab_t, 0),
                               torch.full_like(self.lab_t, 1)).data.cpu().numpy()[0, 0, :, :]

        pre = torch.argmax(self.lab_t_pre, dim=1).data.cpu().numpy()[0, :, :]
        gt = self.lab_t.data.cpu().numpy()[0, 0, :, :]
        acc_real = np.sum((pre == gt) * validmap / np.sum(validmap))

        return task_loss, acc_real

    def optimize_parameters(self,train_or_test='train'):
        self.forward()
        # T2Net
        self.optimizer_T2Net.zero_grad()
        # -----syn2task
        task_loss_syn, self.acc_syn = self.backward_translated2depth()

        self.loss_lab_s = task_loss_syn

        # -------real2task
        task_loss_real, self.acc_real = self.backward_real2depth()
        self.loss_lab_t = task_loss_real
        if (train_or_test == 'train'):
            self.loss_all=self.loss_lab_s+self.loss_lab_t
            self.loss_all.backward()
            self.optimizer_T2Net.step()
