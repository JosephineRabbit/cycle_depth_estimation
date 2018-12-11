from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Helper Functions
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import re
import pdb

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
}

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=False, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU(True)
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
def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']
        l=0
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                l+=1
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        print(l)

   # features = model.features
   # features.block0 = nn.Sequential(features.conv0, features.norm0, features.relu0)

    #features.denseblock1 = nn.Sequential(*list(features.denseblock1))
    #features.transition1 = nn.Sequential(*list(features.transition1)[:-1])

    #features.denseblock2 = nn.Sequential(*list(features.denseblock2))
    #features.transition2 = nn.Sequential(*list(features.transition2)[:-1])

    #features.denseblock3 = nn.Sequential(*list(features.denseblock3))
    #features.transition3 = nn.Sequential(*list(features.transition3)[:-1])

    #features.denseblock4 = nn.Sequential(*(list(features.denseblock4) + [features.norm5]))
    #model.features = features
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        #    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        x = self.features.block0(x)  # 1/4

        x = self.features.denseblock1(x)
        x = self.features.transition1(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 1/8

        x = self.features.denseblock2(x)
        x = self.features.transition2(x)
        outputs.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 1/16

        x = self.features.denseblock3(x)
        x = self.features.transition3(x)
        outputs.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 1/32

        x = self.features.denseblock4(x)
        outputs.append(x)
        return outputs

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

def init_net(net, init_type='normal', init_gain=0.02):
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)

    net = net.cuda()
    net = nn.DataParallel(net)
    init_weights(net, init_type, gain=init_gain)

    return net

class G_1(nn.Module):
    def __init__(self, growth_rate=32, block_config=6,
                     num_init_features=64, bn_size=4, drop_rate=0,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,n_blocks=6,padding_type = 'reflect'):
        assert(n_blocks>0)
        super(G_1, self).__init__()
        self.input_nc = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        i = 0
        num_layers=6
        #for i, num_layers in enumerate(block_config):
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate
        #if i != len(block_config) - 1:
       # trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
       # self.features.add_module('transition%d' % (i + 1), trans)
       # num_features = num_features // 2


        input_nc =  num_features
        model =  [nn.ReflectionPad2d(1),
                  nn.Conv2d(input_nc,ngf, kernel_size=3, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        for i in range(n_blocks):
           # print('++++',i)
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,use_bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        input=self.features(input)
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self ,conv_dim=64, repeat_num=3):
        super(Discriminator, self).__init__()
        layers = []
        curr_dim = conv_dim+1
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout2d(0.2))
            curr_dim = curr_dim * 2

        #kernel_size = int(image_size / np.power(2, repeat_num-1))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, input,sege):

        sege = nn.functional.upsample(input=sege,scale_factor=0.25)
        h = self.main(torch.cat([sege,input],dim=1))

        out_src = self.conv1(h)
        #print(out_src.shape)

        return out_src.squeeze(1)

class ResnetBlock(nn.Module):
    def __init__(self,dim,padding_type,norm_layer,use_dropout,use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self,dim,padding_type,norm_layer,use_dropout,use_bias):
        conv_block = []
        p = 0
        if  padding_type =='reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type=='replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type =='zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented'%padding_type)

        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,padding=p,bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if  padding_type =='reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type=='replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type =='zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented'%padding_type)

        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,padding=p,bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self,input):
        out = input + self.conv_block(input)
        return out

class General_net(nn.Module):
    def __init__(self,mid_nc=1024,num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bn_size=4, drop_rate=0):
        super(General_net, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        num_input_features = num_features
        self.psp = nn.ModuleList()
        num_output_features = mid_nc
        self.psp.append(nn.BatchNorm2d(num_input_features))
        self.psp.append(nn.ReLU(inplace=True))
        self.psp.append(nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=1, stride=1, bias=False))
        self.psp.append(nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=1, stride=1, dilation=1, bias=False))
        self.psp.append(nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=2, stride=1,padding=1, dilation=2,
                                  bias=False))
        self.psp.append(nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=3, stride=1,padding=2, dilation=2,
                                  bias=False))
      #  self.psp.append(nn.Conv2d(num_input_features, num_output_features / 5, kernel_size=3, stride=1, dilation=2,
       #                           bias=False))

        # Final batch norm
        self.psp.append(nn.BatchNorm2d(mid_nc))


    def forward(self,input,R_or_S):
        if R_or_S=='S':
            features=[]
           # print(input.shape)
           # features.append(input)
            features.append(input)
          #  print(input.shape)
            for i in range(len(self.features) - 4):
                # if i < (len(self.features) - 4):
                input = self.features[i + 4].forward(input)
                #         print(i,input.shape)
                if i % 2 == 0 and i != 0:
                    #        print(input.shape)
                    features.append(input)
            input  = self.psp[0](input)
            input = self.psp[1](input)

            input = torch.cat([self.psp[2](input),
                                   self.psp[3](input),
                                   self.psp[4](input),
                                   self.psp[5](input)],1)

            input = (self.psp[6](input))

            return  features,input
        if R_or_S=='R':
            features = []
            input = self.features[0](input)
            input = self.features[1](input)
            input = self.features[2](input)
            input = self.features[3](input)
            features.append(input)
            #print(input.shape)
            for i in range(len(self.features)-4):
                #if i < (len(self.features) - 4):
                input = self.features[i + 4].forward(input)
                #         print(i,input.shape)
                if i % 2 == 0 and i !=0:
                        #        print(input.shape)
                        features.append(input)

            input = self.psp[0](input)
            input = self.psp[1](input)

            input = torch.cat([self.psp[2](input),
                               self.psp[3](input),
                               self.psp[4](input),
                               self.psp[5](input)], 1)
            input = (self.psp[6](input))
            return features, input

        if R_or_S == 'R1':

            input = self.features[0](input)
            input = self.features[1](input)
            input = self.features[2](input)
            input = self.features[3](input)

            return input.detach()

class Discriminator_seg(nn.Module):

    def __init__(self, conv_dim=256, repeat_num=2):
        super(Discriminator_seg, self).__init__()
        layers = []
        curr_dim = conv_dim + 1
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout2d(0.2))
            curr_dim = curr_dim * 2

        # kernel_size = int(image_size / np.power(2, repeat_num-1))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=1, stride=1, bias=False)

    def forward(self,input,sege):
        sege = nn.functional.upsample(input=sege, scale_factor=0.25)
        h = self.main(torch.cat([sege, input], dim=1))

        out_src = self.conv1(h)
        # print(out_src.shape)

        return out_src.squeeze(1)



class SEG(nn.Module):
    def __init__(self,n_cls):
        super(SEG, self).__init__()
        self.trans_0 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.trans_3 = nn.Conv2d(1664,1024,1,1)
        self.trans_2 = nn.Conv2d(1280,512,1,1)
        self.trans_1 = nn.Conv2d(512, 256, 1, 1)
       # self.trans_1 = nn.Conv2d(512,128,1,1)

        self.Up = nn.ModuleList()
        self.Up.append(DeconvBlock(1024 + 1024, 512))
        self.Up.append(DeconvBlock(512 + 512, 256))
        self.Up.append(DeconvBlock(256 + 256, 256))
        #self.Up.append(DeconvBlock(256 , 256))

        self.Up.append(nn.Conv2d(256 , n_cls, 1, 1))
        #self.Dis = nn.Conv2d(256, n_cls, 1, 1)
        self.activation_seg = nn.Sequential(nn.BatchNorm2d(n_cls),
                                            nn.LeakyReLU())
        #self.dis_activation = nn.LeakyReLU()

    def forward(self,features,input):
        features_s = []
       # print(features[0].shape)
        features[0]=self.trans_0(features[0])
        features[1] = self.trans_1(features[1])
        features[2] = self.trans_2(features[2])
      #  features[4] = self.trans_4(features[4])
        features[3] = self.trans_3(features[3])
        S = []
        S.append(input)
       # print(len(features))
        for i in range(len(features)):

            j = len(features) - i - 1
            #print(j, features[j].shape)
            features_s.append(features[j])
        #    print(i, S[i].shape, features[j].shape)

            if i== len(features)-1:
                S.append(self.Up[i](S[i]))
         #       dis = self.Dis(S[len(features)-1])
            else:
                S.append(self.Up[i](torch.cat([S[i], features_s[i]], 1)))
       # print(len(S))
        #S.append(self.Up[len(features)](S[len(features)]))
        S[len(features)] = self.activation_seg(S[len(features)])
        S[len(features)] = nn.functional.upsample(input=S[len(features)],scale_factor=4,mode='nearest')
        #dis = self.dis_activation(dis)

        return S[len(features)],S[len(features)-1]


class DEP(nn.Module):
    def __init__(self):
        super(DEP, self).__init__()
        self.trans_0 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.trans_3 = nn.Conv2d(1664, 1024, 1, 1)
        self.trans_2 = nn.Conv2d(1280, 512, 1, 1)
        self.trans_1 = nn.Conv2d(512, 256, 1, 1)
        # self.trans_1 = nn.Conv2d(512,128,1,1)

        self.Up = nn.ModuleList()
        self.Up.append(DeconvBlock(1024 + 1024, 512))
        self.Up.append(DeconvBlock(512 + 512, 256))
        self.Up.append(DeconvBlock(256 + 256, 256))
        # self.Up.append(DeconvBlock(256 , 256))

        self.Up.append(nn.Conv2d(256, 1, 1, 1))
        # self.Dis = nn.Conv2d(256, n_cls, 1, 1)
        self.activation_seg = nn.LeakyReLU()
        # self.dis_activation = nn.LeakyReLU()

    def forward(self, features, input):
        features_s = []
        # print(features[0].shape)
        features[0] = self.trans_0(features[0])
        features[1] = self.trans_1(features[1])
        features[2] = self.trans_2(features[2])
        #  features[4] = self.trans_4(features[4])
        features[3] = self.trans_3(features[3])
        S = []
        S.append(input)
        # print(len(features))
        for i in range(len(features)):

            j = len(features) - i - 1
          #  print(j, features[j].shape)
            features_s.append(features[j])
            #    print(i, S[i].shape, features[j].shape)

            if i == len(features) - 1:
                S.append(self.Up[i](S[i]))
                #       dis = self.Dis(S[len(features)-1])
            else:
                S.append(self.Up[i](torch.cat([S[i], features_s[i]], 1)))
                # print(len(S))
        # S.append(self.Up[len(features)](S[len(features)]))
        S[len(features)] = self.activation_seg(S[len(features)])
        S[len(features)] = nn.functional.upsample(input=S[len(features)], scale_factor=4, mode='nearest')
        # dis = self.dis_activation(dis)

        return S[len(features)]


def define_G(input_nc, output_nc, netG,
             init_type='normal', init_gain=0.02):

    if netG == '6blocks':
        net = G_1(input_nc, output_nc,  norm_layer=nn.BatchNorm2d,
                  use_dropout=False,n_blocks=6,padding_type = 'reflect')
    elif netG == '3blocks':
        net = G_1(input_nc, output_nc, norm_layer=nn.BatchNorm2d,
                  use_dropout=False,n_blocks=3,padding_type = 'reflect')
    #elif netG == 'unet_128':
     #   net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    #elif netG == 'unet_256':
    #    net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain)

def define_D(init_type='normal', init_gain=0.02):

    net = Discriminator()
    #elif netD == 'n_layers':
    #    net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    #elif netD == 'pixel':
    #    net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    #
    #    raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def pretrain(dens):
    for i, name in enumerate(dens.state_dict()):
        print(name)
        if name in G2_dict:
            print(name)
            G2_dict[name] = dens.state_dict()[name]
            n += 1
        if name in G_dict:
            G_dict[name] = dens.state_dict()[name]
            m += 1

    print(n, m)
    torch.save(G.state_dict(), './G1_model.pth')
    torch.save(G2.state_dict(), './General_model.pth')

if __name__ == '__main__':
    x = torch.Tensor(2, 3, 256, 128).cuda()
    s  = torch.Tensor(2, 1, 256, 128).cuda()
    dens = densenet169(pretrained=True).cuda()
    G = G_1().cuda()

    init_net(G)
    G_dict = G.state_dict()
    G2 = General_net().cuda()
    init_net(G2)
    G2_dict = G2.state_dict()

    D = Discriminator().cuda()
    y = G(x)
    d_pre = D(y,s)
   # print(y.shape)
    Features, Input = G2(y,R_or_S='S')
    for ii,i in enumerate(Features):
        print(ii,i.shape)

    #print(Features.shape,Input.shape)
    SEg = SEG(n_cls=28).cuda()
    z,s = SEg(Features,Input)

    print(F)
    print(z.shape,s.shape)

