import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .encoder_decoder import _UNetEncoder,_UNetDecoder
import torch.nn as nn
from util.util import scale_pyramid
class SegCycle(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','segAreal','segBreal','segAfake','segBfake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A','lab_A','segAreal','segAfake','idt_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B','lab_B','segBreal','segBfake','idt_B']

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','encoderA','encoderB','decoderA','decoderB']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.net_G_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.net_G_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.net_encoderA= _UNetEncoder(input_nc=3).to(opt.gpu_ids[0])
        self.net_encoderB= _UNetEncoder(input_nc=3).to(opt.gpu_ids[0])
        self.net_decoderA= _UNetDecoder(output_nc=22).to(opt.gpu_ids[0])
        self.net_decoderB= _UNetDecoder(output_nc=28).to(opt.gpu_ids[0])
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.net_D_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.net_D_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_G_A.parameters(), self.net_G_B.parameters(),
                                                                self.net_encoderA.parameters(),self.net_encoderB.parameters(),
                                                                self.net_decoderA.parameters(),self.net_decoderB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['img_source'].cuda()
        self.real_B = input['img_target'].cuda()
        self.lab_A = input['lab_source'].cuda()
        self.lab_B = input['lab_target'].cuda()

    def forward(self):
        self.fake_B = self.net_G_A(self.real_A)
        self.rec_A = self.net_G_B(self.fake_B)

        self.fake_A = self.net_G_B(self.real_B)
        self.rec_B = self.net_G_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        return loss_D

    def Seg_basic(self, encoder, decoder, input,gt):
        # Real
        embedding=encoder(input)
        output=decoder(embedding)
        criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=255).cuda()
        task_loss = criterion(output[-1], gt.squeeze(1))
        return task_loss,output

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.net_D_A, self.real_B, fake_B)
        return self.loss_D_A

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.net_D_B, self.real_A, fake_A)
        return self.loss_D_B

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.net_G_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.net_G_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_segAreal,self.segAreal = self.Seg_basic(encoder=self.net_encoderA,decoder=self.net_decoderA,input=self.real_A,
                                            gt=self.lab_A)
        self.loss_segAfake,self.segAfake = self.Seg_basic(encoder=self.net_encoderB, decoder=self.net_decoderA, input=self.fake_B,
                                            gt=self.lab_A)
        self.loss_segBreal,self.segBreal= self.Seg_basic(encoder=self.net_encoderB, decoder=self.net_decoderB, input=self.real_B,
                                            gt=self.lab_B)
        self.loss_segBfake,self.segBfake = self.Seg_basic(encoder=self.net_encoderA, decoder=self.net_decoderB, input=self.fake_A,
                                            gt=self.lab_B)

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.net_D_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.net_D_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B+\
            self.loss_segAfake+self.loss_segAreal+self.loss_segBfake+self.loss_segBreal
        return self.loss_G
    def optimize_parameters(self,train_or_test):

        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.net_D_A, self.net_D_B], False)
        self.set_requires_grad([self.net_G_B, self.net_G_A,self.net_encoderB,self.net_encoderA,self.net_decoderA,self.net_decoderB], True)
        self.optimizer_G.zero_grad()
        self.loss_G=self.backward_G()
        if(train_or_test=='train'):
            self.loss_G.backward()
            self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.net_D_A, self.net_D_B], True)
        self.set_requires_grad([self.net_G_B, self.net_G_A,self.net_encoderB,self.net_encoderA,self.net_decoderA,self.net_decoderB], False)
        self.optimizer_D.zero_grad()
        self.loss_D_A=self.backward_D_A()
        self.loss_D_B=self.backward_D_B()

        if(train_or_test=='train'):
            self.loss_D_A.backward()
            self.loss_D_B.backward()
            self.optimizer_D.step()
