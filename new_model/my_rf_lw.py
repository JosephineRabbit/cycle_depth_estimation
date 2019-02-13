import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from new_model.utils.layer_factory import convbnrelu,conv1x1, conv3x3, CRPBlock,PSPBlock,ATBlock

data_info = {
    7 : 'Person',
    21: 'VOC',
    40: 'NYU',
    60: 'Context'
    }

models_urls = {
    '50_person'  : 'https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download',
    '101_person' : 'https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download',
    '152_person' : 'https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download',

    '50_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/2E1KrdF2Rfc5khB/download',
    '101_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download',
    '152_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download',

    '50_nyu'     : 'https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download',
    '101_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download',
    '152_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download',

    '101_context': 'https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download',
    '152_context': 'https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download',

    '50_imagenet' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}



class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=False, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        #self.drop = torch.nn.Dropout(0.05)
        self.relu = torch.nn.LeakyReLU(0.02)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class DCGAN_D(torch.nn.Module):
	def __init__(self,D_h_size=64):
		super(DCGAN_D, self).__init__()
		main = torch.nn.Sequential()

		### Start block
		# Size = n_colors x image_size x image_size
		main.add_module('Start-Conv2d', torch.nn.Conv2d(2, D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
		main.add_module('Start-SELU', torch.nn.SELU(inplace=True))

		#image_size_new = param.image_size // 2
		# Size = D_h_size x image_size/2 x image_size/2

		### Middle block (Done until we reach ? x 4 x 4)
		mult = 1
		i = 0
		for i in range(5):
			main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(D_h_size * mult, D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))

			mult *= 2
			#i += 1

		### End block
		# Size = (D_h_size * mult) x 4 x 4
		main.add_module('End-Conv2d', torch.nn.Conv2d(D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
		main.add_module('End-Sigmoid', torch.nn.Sigmoid())
		# Size = 1 x 1 x 1 (Is a real cat or not?)
		self.main = main

	def forward(self, input):
		output = torch.nn.parallel.data_parallel(self.main, input, range(2))

		return output.view(-1)

class depth_block(nn.Module):
    def __init__(self,in_c):
        super(depth_block, self).__init__()
        self.upconv = nn.ModuleList()
        self.depth_out = nn.ModuleList()
        self.attention_bs =nn.ModuleList()
        for i in range(4):
            self.upconv.append(nn.Sequential(nn.ConvTranspose2d(in_c,int(in_c/2),4,2,padding=1) , nn.LeakyReLU(0.02)
            ,nn.BatchNorm2d(int(in_c/2)),
            nn.Conv2d(int(in_c / 2), int(in_c / 2), 1, 1),nn.ReLU6(),nn.BatchNorm2d(int(in_c/2))
                                             ))

            self.depth_out.append(nn.Sequential(nn.Conv2d(int(in_c/2),1,3,1,padding=1),nn.Tanh()
                                                ))

            self.attention_bs.append(nn.Sequential(nn.Conv2d(in_c,int(in_c/2),3,2,padding=1) , nn.ReLU6()
            ,nn.BatchNorm2d(int(in_c/2)),

            nn.AdaptiveAvgPool2d(1)))
        #self.depth_out.append(nn.Sequential(nn.Conv2d(int(in_c / 2), 1, 1, 1), nn.BatchNorm2d(1), nn.Tanh()))

        self.at_act = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(int(in_c*2),int(in_c/2),3,1,padding=1),
                                     nn.LeakyReLU(0.02),
                                  nn.BatchNorm2d(in_c//2),

                                                 )



        #self.s_econv = nn.Sequential(
         #                            nn.Conv2d(int(in_c/2),1,3,1,padding=1),nn.BatchNorm2d(1),
          #                           nn.Sigmoid())
        self.depconv = nn.Sequential(nn.Conv2d(int(in_c/2),1,3,stride=1,padding=1),#nn.BatchNorm2d(1),
                                     nn.Tanh()
                                     )

    def forward(self,in_f):
        dep_o = []
        features = []
        at = []
        out_f = []

        for i in range(4):
            features.append(self.upconv[i](in_f))
            dep_o.append(self.depth_out[i](features[i]))
            at.append(self.attention_bs[i](in_f))

            out_f.append(torch.mul(self.at_act(at[i]),features[i])+features[i])
        F = torch.cat([out_f[0],out_f[1],out_f[2],out_f[3]],1)
        F = self.conv(F)


        dep_1 = self.depconv(F)

        return dep_o,dep_1

class ResNetLW(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetLW, self).__init__()
       # self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.ins_layer1_s = self._make_instancelayer(planes=256)
        self.ins_layer1_r = self._make_instancelayer(planes=256)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.ins_layer2_s = self._make_instancelayer(planes=512)
        self.ins_layer2_r = self._make_instancelayer(planes=512)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.ins_layer3_s = self._make_instancelayer(planes=1024)
        self.ins_layer3_r = self._make_instancelayer(planes=1024)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.ins_layer4_s = self._make_instancelayer(planes=2048)
        self.ins_layer4_r = self._make_instancelayer(planes=2048)


        self.p_ims1d2_outl1_dimred_ = convbnrelu(2048,1024,1)
       # self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.psp4 = self._make_psp(1024, 1024)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(1024,512, bias=False)
        self.p_ims1d2_outl2_dimred_= convbnrelu(1024, 512, 1)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(512, 512, bias=False)
       # self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.psp3 = self._make_psp(512,512)
        self.CAT3 = ATBlock(512,512)

        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl3_dimred_ = convbnrelu(512, 256,1)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        #self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.psp2 = self._make_psp(256, 256)
        self.CAT2 = ATBlock(256, 256)

        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred_ = convbnrelu(256, 256,1)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
       # self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.psp1 = self._make_psp(256, 256)
        self.CAT1 = ATBlock(256, 256)
        self.dep=depth_block(256)




        # self.out_conv = nn.Conv2d(256, 1, kernel_size=3, stride=1,
        #                           padding=1, bias=True)

    def _make_psp(self,inplanes,out_planes):
        layers  = [PSPBlock(inplanes,out_planes,4)]
        return  nn.Sequential(*layers)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_instancelayer(self, planes, stride=1):
        ins_layer = nn.Sequential(
                nn.Conv2d(planes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm2d(planes),
               # nn.Conv2d(planes, planes,
                #      kernel_size=1, stride=1, bias=False),
               # nn.InstanceNorm2d(planes)
         )
        return ins_layer

    def forward(self, x,type='real'):
        if type=='real':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            l1 = self.layer1(x)
            l1 = self.ins_layer1_r(l1)
           # print('2',l1.shape)
            l2 = self.layer2(l1)
            l2 = self.ins_layer2_r(l2)
           # print('3',l2.shape)
            l3 = self.layer3(l2)
            l3 = self.ins_layer3_r(l3)
           # print('4', l3.shape)
            l4 = self.layer4(l3)
            l4 = self.ins_layer4_r(l4)
           # print('5', l4.shape)

          #  l4 = self.do(l4)
          #  l3 = self.do(l3)
            #print(l3.shape)

            x4 = self.p_ims1d2_outl1_dimred_(l4)
            x4 = self.relu(x4)
        #    print(x4.shape)
            #x4 = self.mflow_conv_g1_pool(x4)#crp
            x4 = self.psp4(x4)
            x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)#rcu*3
            #print(x4.shape)
            x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)
          #  print('de_1',x4.shape)

            #print(l3.shape)
            x3 = self.p_ims1d2_outl2_dimred_(l3)
            x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
            x3 = self.CAT3(x3,x4)


            #x3 = F.relu(x3)
            x3 = self.psp3(x3)
           # x3 = self.mflow_conv_g2_pool(x3)


            x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
            x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
         #   print('de_2', x3.shape)

            x2 = self.p_ims1d2_outl3_dimred_(l2)
            x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
            x2 = self.CAT2(x2,x3)
            x2 = self.psp2(x2)
            x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
            x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)
        #    print('de_3', x2.shape)

            x1 = self.p_ims1d2_outl4_dimred_(l1)
            x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
         #   x1 = x1 + x2
            x1 = self.CAT1(x1,x2)
            x1 = self.psp1(x1)
        #    x1 = self.mflow_conv_g4_pool(x1)
       #     print('de_4', x1.shape)

            outs,pred_d = self.dep(x1)

            return outs,pred_d,(x4,x3,x2,x1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # print('1',x.shape)

            l1 = self.layer1(x)
            l1 = self.ins_layer1_s(l1)
            # print('2',l1.shape)
            l2 = self.layer2(l1)
            l2 = self.ins_layer2_s(l2)
            # print('3',l2.shape)
            l3 = self.layer3(l2)
            l3 = self.ins_layer3_s(l3)
            # print('4', l3.shape)
            l4 = self.layer4(l3)
            l4 = self.ins_layer4_s(l4)
            # print('5', l4.shape)

            #  l4 = self.do(l4)
            #  l3 = self.do(l3)
            # print(l3.shape)

            x4 = self.p_ims1d2_outl1_dimred_(l4)
            x4 = self.relu(x4)
            #    print(x4.shape)
            # x4 = self.mflow_conv_g1_pool(x4)#crp
            x4 = self.psp4(x4)
            x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)  # rcu*3
            # print(x4.shape)
            x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)
            #  print('de_1',x4.shape)

            # print(l3.shape)
            x3 = self.p_ims1d2_outl2_dimred_(l3)
            x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
            x3 = self.CAT3(x3, x4)

            # x3 = F.relu(x3)
            x3 = self.psp3(x3)
            # x3 = self.mflow_conv_g2_pool(x3)


            x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
            x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
            #   print('de_2', x3.shape)

            x2 = self.p_ims1d2_outl3_dimred_(l2)
            x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
            x2 = self.CAT2(x2, x3)
            x2 = self.psp2(x2)
            x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
            x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)
            #    print('de_3', x2.shape)

            x1 = self.p_ims1d2_outl4_dimred_(l1)
            x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
            #   x1 = x1 + x2
            x1 = self.CAT1(x1, x2)
            x1 = self.psp1(x1)
            #    x1 = self.mflow_conv_g4_pool(x1)
            #     print('de_4', x1.shape)

            outs, pred_d = self.dep(x1)

            return outs, pred_d, (x4, x3, x2, x1)

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

def init_net(net, init_type='kaiming', init_gain=0.02):
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)

    net = net.cuda()
  #  net = nn.DataParallel(net)
    init_weights(net, init_type, gain=init_gain)

    return net
class segd(nn.Module):
    def __init__(self,n_cls,up_scale,init_channel=256):
        super(segd, self).__init__()
        self.Up = nn.ModuleList()
        init_channel=init_channel
        for i in range(up_scale):
            self.Up.append(DeconvBlock(init_channel, int(init_channel/2)))
            init_channel=int(init_channel/2)


        self.Up.append(nn.Conv2d(init_channel, n_cls+1, 1, 1))
        #self.activation_seg = nn.Sequential(nn.BatchNorm2d(n_cls),
         #                                   nn.LeakyReLU())

    def forward(self, input):
        S = []
        S.append(input)
        # print(len(features))
        for i in range(len(self.Up)):
            S.append(self.Up[i](S[i]))

        S[len(self.Up)] = S[len(self.Up)]

        return S[len(self.Up)]


def rf_lw101():
    model = ResNetLW(Bottleneck, [3, 4, 23, 3])
    return model

class seg_gan_loss():
    def __init__(self):
        super(seg_gan_loss, self).__init__()

        self.loss = nn.MSELoss()
        self.classify = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=255)



    def __call__(self, input,label, target_is_real):
        if target_is_real:
            return self.classify(input,label)
        else:
            new_label=torch.tensor(28).cuda().expand_as(label)
            return self.classify(input,new_label)

if __name__ == '__main__':

    x = torch.Tensor(2, 3, 576,192).cuda()

    s = torch.zeros(2, 2, 576,192).cuda()
    net = rf_lw101().cuda()
    net = init_net(net)
    G_dict = net.state_dict()

    ori_dict = torch.load('/home/dut-ai/Documents/light-weight-refinenet-master/models/rf_lw101_img.pth')
    segd8 = segd(n_cls=40,up_scale=3,init_channel=512).cuda()
    segd4 = segd(n_cls=40,up_scale=2).cuda()
    segd2 = segd(n_cls=40,up_scale=1).cuda()
    segd2_0 = segd(n_cls=40,up_scale=1).cuda()


    m=0
    n=0
    for i,name in enumerate(ori_dict):


        if name in G_dict:

            #print(name)
            if G_dict[name].shape == ori_dict[name].shape:
                print(name)
                G_dict[name] = ori_dict[name]
                n += 1

        m += 1
    print(n,m)
    torch.save(G_dict, './my_ins_res101_model.pth')
    print(net)
    outs,pred_d,features = net(x)
    dis1 = segd8(features[0])
    dis2 = segd4(features[1])
    dis3 = segd2(features[2])
    dis4 = segd2_0(features[3])
    print(dis1.shape,dis2.shape,dis3.shape,dis4.shape)
    #print(out.shape)
    for i in outs:
        print(i.shape)
    print(pred_d.shape)