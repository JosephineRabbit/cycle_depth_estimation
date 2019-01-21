from torch.nn import init
import torch.autograd as autograd
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


class G_1(nn.Module):
    def __init__(self, growth_rate=32, block_config=6,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
        assert (n_blocks > 0)
        super(G_1, self).__init__()
        self.input_nc = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=False)),
            #  ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        num_layers = block_config
        i = 0
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate

        input_nc = num_features
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(False)]

        for i in range(n_blocks):
            # print('++++',i)
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        input = self.features(input)
        return self.model(input)

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

def densenet169(pretrained=False, **kwargs):
    """Densenet-169 model from
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

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
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

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
            ('relu0', nn.ReLU(inplace=False)),
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

class Discriminator(nn.Module):
    def __init__(self , curr_dim = 2, conv_dim=32, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []

        layers.append(nn.Conv2d(curr_dim, conv_dim, kernel_size=3, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        #kernel_size = int(image_size / np.power(2, repeat_num-1))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, input):

       # sege = nn.functional.upsample(input=sege,scale_factor=0.5)
        h = self.main(input)
        out_src = self.conv1(h)
        out_src = nn.LeakyReLU()(out_src)
        return out_src.squeeze(1)

class ResnetBlock(nn.Module):
    def __init__(self,in_dim,padding_type,norm_layer,use_dropout,use_bias):
        super(ResnetBlock, self).__init__()
        self.conv0_block = self.build_conv0_block(in_dim, padding_type, norm_layer, use_dropout, use_bias)
        self.conv1_block = self.build_conv1_block(in_dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv0_block(self, in_dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        dim = in_dim
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(0)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(0)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim , kernel_size=1, dilation=2,padding=p,bias=use_bias),
                       norm_layer(dim)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def build_conv1_block(self,in_dim,padding_type,norm_layer,use_dropout,use_bias):
        conv_block = []

        dim = in_dim
        p = 0

        conv_block += [nn.ReflectionPad2d(1)]


        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(False)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]


        return nn.Sequential(*conv_block)

    def forward(self,input):
        #print(self.conv0_block(input).shape,self.conv1_block(input).shape)
        out = input +self.conv0_block(input)+self.conv1_block(input)
        return out

class Seg2Feature(nn.Module):
    def __init__(self):
        super(Seg2Feature, self).__init__()

class _pspTrans(nn.Module):
    def __init__(self,num_input_features):
        super(_pspTrans,self).__init__()
        num_output_features = int(num_input_features/4)
        self.trans = nn.ModuleList()
        self.trans.append(nn.BatchNorm2d(num_input_features))
        self.trans.append(nn.ReLU(inplace=False))
        self.trans.append(nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.trans.append(nn.Conv2d(num_input_features, num_output_features,
                                    kernel_size=3, stride=1,padding=1, bias=False))
        self.trans.append(nn.AvgPool2d(kernel_size=2, stride=2))
    def forward(self,input):
        input = self.trans[0](input)
        input = self.trans[1](input)
        input = torch.cat([self.trans[2](input),self.trans[3](input)],1)
        input = self.trans[4](input)
        return input




class General_net(nn.Module):
    def __init__(self,mid_nc=1024,num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bn_size=4, drop_rate=0):
        super(General_net, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
       #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        self.PSP = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
           #     print(i,num_features)
                trans_ = _pspTrans(num_features)
                self.PSP.append(trans_)
         #       self.PSP.append(_pspTransition(num_input_features=num_features))
                num_features = num_features // 2

        num_input_features = num_features
        self.psp = nn.ModuleList()
        num_output_features = mid_nc
        self.psp.append(nn.BatchNorm2d(num_input_features))
        self.psp.append(nn.ReLU(inplace=True))
        self.psp.append(
            nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=1, stride=1, bias=False))
        self.psp.append(nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=1, stride=1, dilation=1,
                                  bias=False))
        self.psp.append(
            nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=2, stride=1, padding=1, dilation=2,
                      bias=False))
        self.psp.append(
            nn.Conv2d(num_input_features, int(num_output_features / 4), kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=False))

        self.psp.append(nn.BatchNorm2d(mid_nc))

    def forward(self,input,type):
        if type == 'R':
            features = []
          #  print(len(self.PSP),len(self.features))

            for i in range(3):
                input = self.features[i](input)
            #    print(i,input.shape)
            i = 0
            for i in range(len(self.features)-3):
              #  print(i,fe)

                j = i+3

                input = self.features[j].forward(input)
              #  print(i, input.shape)
                features.append(input.detach())

                if i<3:
                    input = self.PSP[i](input)


                #print(i,input.shape)

            input = self.psp[0](input)
            input = self.psp[1](input)

            input = torch.cat([self.psp[2](input),
                               self.psp[3](input),
                               self.psp[4](input),
                               self.psp[5](input)], 1)

            input = (self.psp[6](input))


            return input,features

        if type == 'S':
            features = []
            #  print(len(self.PSP),len(self.features))


            for i in range(len(self.features) - 3):
                #  print(i,fe)
                j = i + 3

                input = self.features[j].forward(input)
                #  print(i, input.shape)
                features.append(input.detach())

                if i < 3:
                    input = self.PSP[i](input)


                    # print(i,input.shape)

            input = self.psp[0](input)
            input = self.psp[1](input)

            input = torch.cat([self.psp[2](input),
                               self.psp[3](input),
                               self.psp[4](input),
                               self.psp[5](input)], 1)

            input = (self.psp[6](input))

            return input, features
def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

class _MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=1, norm='batch', activation='PReLU'):
        super(_MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D


        for i in range(num_D):
            netD = _Discriminator(input_nc, ndf, n_layers, norm, activation)
            setattr(self, 'scale'+str(i), netD)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, 'scale'+str(i))
            output = netD.forward(input)
            result.append(output)
            if i != (self.num_D-1):
                input = self.downsample(input)
        return result


class _Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', activation='PReLU'):
        super(_Discriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nonlinearity,
        ]

        nf_mult=1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            model += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.PReLU(),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nonlinearity,
            nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class _FeatureDiscriminator(nn.Module):
    def __init__(self, input_nc, n_layers=2, norm='batch',size=(40,12), activation='PReLU'):
        super(_FeatureDiscriminator, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.size = size
        self.in_c = input_nc



        model = [
            nn.Linear(input_nc * int(size[0]) * int(size[1]), input_nc),
            nn.PReLU(),
        ]

        for i in range(1, n_layers):
            model +=[
                nn.Linear(input_nc, input_nc),
                nn.PReLU()
            ]

        model +=[nn.Linear(input_nc, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        input = input.view(-1, self.in_c * self.size[0] *self.size[1])
        output = self.model(input)

        return output


class attentionblock(nn.Module):
    def __init__(self,flag,in_c,out_c):
        super(attentionblock, self).__init__()
        if flag:
            self.atb= nn.Sequential(nn.Conv2d(in_c,out_c,3,2,padding=1)
            ,nn.BatchNorm2d(out_c)
            , nn.LeakyReLU(0.02),
            # nn.Conv2d(in_c, out_c, 3, 2, padding=1)
            # , nn.BatchNorm2d(out_c)
            # , nn.LeakyReLU(0.02)
            nn.AdaptiveAvgPool2d(1))
        else:
            self.atb = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 2, padding=1)
                                     , nn.BatchNorm2d(out_c)
                                     , nn.LeakyReLU(0.02)
                                     , nn.AdaptiveAvgPool2d(1))

    def forward(self,input):
        out = self.atb(input)
        return out

class G_side(nn.Module):
    def __init__(self,side_c,df_c,f_size):
        super(G_side, self).__init__()


        self.attention_bs=nn.Sequential(nn.Conv2d(df_c,df_c,3,2,padding=1)
            ,nn.BatchNorm2d(df_c)
            , nn.LeakyReLU(0.02)
            ,nn.AdaptiveAvgPool2d(1))

        self.at_act = nn.Sigmoid()
        self.side_conv = nn.Sequential(nn.Conv2d(side_c,df_c,3,1,padding=1),nn.LeakyReLU(0.02),nn.BatchNorm2d(df_c),
                                       nn.Conv2d(df_c,df_c,1,1),nn.LeakyReLU(0.02),nn.BatchNorm2d(df_c))
        self.conv = nn.Sequential(nn.Conv2d(df_c,int(df_c/2),3,1,padding=1),nn.LeakyReLU(0.02),nn.BatchNorm2d(int(df_c/2)),
                                  nn.Conv2d(int(df_c/2),int(df_c/2),1,1),nn.LeakyReLU(0.02),nn.BatchNorm2d(int(df_c/2)),nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self,s_feature,d_featuers):
        attention = self.attention_bs(d_featuers)
        attention_= self.at_act(attention)

       # attention = self.up(attention)
        s_f = self.side_conv(s_feature)
        s_f = torch.mul(attention_,s_f)
    #    print('d_f',d_featuers.shape)

        #print(d_featuers.shape,s_f.shape)
        out = d_featuers+s_f
        out = self.conv(out)
        return out



class depth_block(nn.Module):
    def __init__(self,in_c):
        super(depth_block, self).__init__()
        self.upconv = nn.ModuleList()
        self.depth_out = nn.ModuleList()
        self.attention_bs =nn.ModuleList()
        for i in range(4):
            self.upconv.append(nn.Sequential(nn.ConvTranspose2d(in_c,int(in_c/2),4,2,padding=1) , nn.LeakyReLU(0.02)
            ,nn.BatchNorm2d(int(in_c/2)),
            nn.Conv2d(int(in_c / 2), int(in_c / 2), 3, 1, padding=1),nn.LeakyReLU(0.02),nn.BatchNorm2d(int(in_c/2))
                                             ))

            self.depth_out.append(nn.Sequential(nn.Conv2d(int(in_c/2),1,3,1,padding=1),nn.Tanh()
                                                ))

            self.attention_bs.append(nn.Sequential(nn.Conv2d(in_c,int(in_c/2),3,2,padding=1) , nn.LeakyReLU(0.02)
            ,nn.BatchNorm2d(int(in_c/2)),

            nn.Conv2d(int(in_c / 2), int(in_c / 2), 3, 2, padding=1),nn.LeakyReLU(0.02)
            ,nn.AdaptiveAvgPool2d(1)))
        #self.depth_out.append(nn.Sequential(nn.Conv2d(int(in_c / 2), 1, 1, 1), nn.BatchNorm2d(1), nn.Tanh()))

        self.at_act = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(int(in_c*2),int(in_c),3,1,padding=1),
                                     nn.LeakyReLU(0.02),
                                  nn.BatchNorm2d(in_c),
                                  nn.Conv2d(int(in_c), int(in_c / 2), 3, 1, padding=1), nn.LeakyReLU(0.02),nn.BatchNorm2d(int(in_c / 2))
                                                 )


        self.depconv = nn.Sequential(nn.Conv2d(int(in_c/2),1,3,stride=1,padding=1),nn.BatchNorm2d(1),
                                     #nn.Tanh()
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
            #print('att',at[i].shape)
            out_f.append(torch.mul(self.at_act(at[i]),features[i])+features[i])
        F = torch.cat([out_f[0],out_f[1],out_f[2],out_f[3]],1)
        F = self.conv(F)

        dep_1 = self.depconv(F)

        return  dep_o,dep_1

class seg_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(seg_block, self).__init__()
        self.deconv = nn.Sequential(nn.Conv2d(int(in_c),int(in_c),3,1,padding=1),
                                     nn.LeakyReLU(0.02),
                                  nn.BatchNorm2d(in_c),nn.Conv2d(in_c, out_c, 1, 1),nn.UpsamplingBilinear2d(scale_factor=2))
        self.bn = torch.nn.BatchNorm2d(out_c)
       # self.result = nn.Conv2d(out_c,28,1,1)
        self.lru = nn.LeakyReLU(0.02)
       # self.act = nn.Tanh()


        # self.drop = torch.nn.Dropout(0.05)
     #   self.relu = torch.nn.LeakyReLU(0.02)

    def forward(self, x):

        out = self.lru(self.bn(self.deconv(x)))
      #  res = self.lru(self.result(out))

        return out




class R_dep(nn.Module):
    def __init__(self):
        super(R_dep, self).__init__()
        self.up0 = DeconvBlock(1024,512)
        self.AT = nn.ModuleList()
        # self.Dep = nn.ModuleList()
        # self.Seg = nn.ModuleList()

        gs0 = G_side(side_c = 1664, df_c = 1024,f_size= 40)
        gs1 = G_side(side_c = 1280, df_c = 512,f_size = 80)
        gs2 = G_side(side_c = 512, df_c = 256,f_size = 160)
        self.seg_d = seg_block(in_c=128,out_c=28)
        self.dep = depth_block(in_c=128)
        #gs3 = G_side(side_c = 256, df_c = 128, f_size = 320)
        #dep0 = depth_block(in_c = 1024,f_size=40)
        #dep1 = depth_block(in_c= 512,f_size=80)
        #dep2 = depth_block(in_c = 256,f_size=160)
        #dep3 = depth_block(in_c=128,f_size=320)
        #seg0 = seg_block(1024,512)
        # seg1 = seg_block(1024,256)
        # seg2 = seg_block(512,128)
        # seg3 = seg_block(256,64)
        self.AT.append(gs0)
        self.AT.append(gs1)
        self.AT.append(gs2)
        # self.AT.append(gs3)
        #
        # self.Dep.append(dep0)
        # self.Dep.append(dep1)
        # self.Dep.append(dep2)
        # self.Dep.append(dep3)
        #
        #
        #
        # self.Seg.append(seg0)
        # self.Seg.append(seg1)
        # self.Seg.append(seg2)
        # self.Seg.append(seg3)

        self.dep_out = nn.Conv2d(64,1,1,1)

        self.norm = nn.BatchNorm2d(1)
        #self.act = nn.Tanh()

    def forward(self,s_features,d_feature):
        out0=self.AT[0](s_features[3],d_feature)
        # seg_f5,seg80 =self.Seg[0](out0)
        #
        # dep_f1, dep_80,s_e80 = self.Dep[0](out0)
        # Dep_80 = torch.cat([nn.UpsamplingBilinear2d(scale_factor=8)(dep_80[0]),
        #                     nn.UpsamplingBilinear2d(scale_factor=8)(dep_80[1]),
        #                     nn.UpsamplingBilinear2d(scale_factor=8)(dep_80[2]),
        #                     nn.UpsamplingBilinear2d(scale_factor=8)(dep_80[3])],1)
        # Seg_80 = nn.UpsamplingNearest2d(scale_factor=8)(seg80)
        # seg_f4, seg160 = self.Seg[1](torch.cat([seg_f5, dep_f1],1))

        out1 = self.AT[1](s_features[2], out0)
    #    out1.backward()
        print('---------------')
        # dep_f2, dep_160,s_e160 = self.Dep[1](out1)
        # Dep_160 = torch.cat([nn.UpsamplingBilinear2d(scale_factor=4)(dep_160[0]),
        #                     nn.UpsamplingBilinear2d(scale_factor=4)(dep_160[1]),
        #                     nn.UpsamplingBilinear2d(scale_factor=4)(dep_160[2]),
        #                     nn.UpsamplingBilinear2d(scale_factor=4)(dep_160[3])], 1)
        # S_e80 = nn.UpsamplingBilinear2d(scale_factor=8)(s_e80)
        # S_e160 = nn.UpsamplingBilinear2d(scale_factor=4)(s_e160)
        # seg_f3, seg320 = self.Seg[2](torch.cat([seg_f4, dep_f2], 1))
        # Seg_160 = nn.UpsamplingNearest2d(scale_factor=4)(seg160)
        # Seg_320 = nn.UpsamplingNearest2d(scale_factor=2)(seg320)

        out2 = self.AT[2](s_features[1],out1)
        sd = self.seg_d(out2)
        de = self.dep(out2)
        # dep_f3, dep_320,s_e320 = self.Dep[2](out2)
        # Dep_320 = torch.cat([nn.UpsamplingBilinear2d(scale_factor=2)(dep_320[0]),
        #                     nn.UpsamplingBilinear2d(scale_factor=2)(dep_320[1]),
        #                     nn.UpsamplingBilinear2d(scale_factor=2)(dep_320[2]),
        #                     nn.UpsamplingBilinear2d(scale_factor=2)(dep_320[3])], 1)
        # S_e320 = nn.UpsamplingBilinear2d(scale_factor=2)(s_e320)
        # seg_f4, seg640 = self.Seg[3](torch.cat([seg_f3, dep_f3], 1))
        #
        # out3 = self.AT[3](s_features[0], dep_f3)
        # dep_f4, dep_640,s_e640 = self.Dep[3](out3)
        # Dep_640 = torch.cat([dep_640[0],
        #                      dep_640[1],
        #                      dep_640[2],
        #                      dep_640[3]], 1)

        return (out0,out1,out2),sd,de


def ordimat(bs,n, m):
    nlist = range(0, n)
    nlist = np.tile(nlist, (m, 1))
    nlist = np.transpose(nlist)

    mlist = range(0, m)
    m = np.tile(mlist, (n, 1))
   # M = np.zeros((2, n, m))
    M=[nlist,m]

    mm = torch.tensor(M).cuda().float()
    mm = mm.unsqueeze(0).repeat(bs, 1, 1, 1)


    return mm



class Discriminator2_seg(nn.Module):

    def __init__(self, conv_dim=1024, repeat_num=3):
        super(Discriminator2_seg, self).__init__()

        layers = []
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim,int(curr_dim / 2), kernel_size=1, stride=1, padding=0))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.BatchNorm2d(int(curr_dim / 2)))
            curr_dim = int(curr_dim / 2)

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=2, padding=1,bias=False)

    def forward(self,input):
        #sege = nn.functional.upsample(input=sege, scale_factor=0.25)
        h = self.main(input)
        out_src = self.conv1(h)
        out_src = nn.Sigmoid()(out_src)

        return out_src.squeeze(1)

class SEG(nn.Module):
    def __init__(self,n_cls):
        super(SEG, self).__init__()

        self.Up = nn.ModuleList()
        self.Up.append(DeconvBlock(1024 , 512))
        self.Up.append(DeconvBlock( 512, 256))
        self.Up.append(DeconvBlock(256, 128))
        self.Up.append(DeconvBlock(128,64))

        self.Up.append(nn.Conv2d(64, n_cls, 1, 1))
        self.activation_seg = nn.Sequential(nn.BatchNorm2d(n_cls),
                                            nn.LeakyReLU())

    def forward(self,input):

        S = []
        S.append(input)
       # print(len(features))
        for i in range(len(self.Up)):
           S.append(self.Up[i](S[i]))

        S[len(self.Up)] = self.activation_seg(S[len(self.Up)])

        return S[len(self.Up)],S[len(self.Up)-5]

class DEP(nn.Module):
    def __init__(self):
        super(DEP, self).__init__()

        self.Up = nn.ModuleList()
        self.Up.append(DeconvBlock(1024+2, 512))
        self.Up.append(DeconvBlock(512+2,256))
        self.Up.append(DeconvBlock(256 + 2, 128))
        self.Up.append(DeconvBlock(128+ 2, 64))

        self.Up.append(nn.Conv2d(64,1, 1, 1))
       # self.activation_seg =nn.Tanh()

    def forward(self, input):
        features_s = []
        Ord=[]
        S = []
        S.append(input)
        # print(len(features))
        for i in range(len(self.Up)):
            if i != (len(self.Up)-1):
                Ord.append(ordimat(S[i].shape[0], S[i].shape[2], S[i].shape[3]))
                S.append(self.Up[i](torch.cat([S[i],Ord[i]],dim=1)))
            else:
                S.append(self.Up[i](S[i]))
            # print(len(S))
        # S.append(self.Up[len(features)](S[len(features)]))
        #S[len(self.Up)] = self.activation_seg(S[len(self.Up)])

        return S[len(self.Up)]




class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())

        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



class BCEDepLoss(nn.Module):
    def __init__(self):
        super(BCEDepLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.L1 = nn.L1Loss()

    def __call__(self, input,target,o_m,z_m):
        target_ = target.contiguous()

        return self.bce((input+1)/2*o_m,(target_+1)/2*o_m)+self.bce((input+1)/2*z_m,(target_+1)/2*z_m)+50*self.L1(input,target_)



def pretrain(dens):

    m, n = 0, 0
    for i, name in enumerate(dens.state_dict()):
        print(name)

        if name in G_dict:
            G_dict[name] = dens.state_dict()[name]
            m += 1

    print(n, m)
    torch.save(G1.state_dict(), './G_1_model.pth')

def get_masks(target):
    o_m = target.clone()
    o_m.requires_grad = False
    z_m = target.clone()
    z_m.requires_grad = False

    o_m[o_m != 1] = 0
    z_m[z_m != -1] = 0
    z_m[z_m == -1] = 1
    return o_m,z_m

if __name__ == '__main__':

    x = torch.Tensor(2, 3, 576,192).cuda()

    s = torch.zeros(2, 2, 576,192).cuda()
    o_m,z_m = get_masks(s)
    dens = densenet169(pretrained=True).cuda()
    G = General_net().cuda()
    R_D = R_dep().cuda()
    #R_D = nn.DataParallel(R_D)
    G1 = G_1().cuda()
    bce = BCEDepLoss()
    #G = nn.DataParallel(G)
    #G1 = nn.DataParallel(G1)
    G1_dict = G1.state_dict()
    op = torch.optim.Adam(R_D.parameters(),
                          0.0001, betas=(0.5, 0.999)
                          )


    y  = G1(x)

    print('y',y.shape)


    syn_features1,features = G(input=y,type='S')
    #for i in range(len(features)):
     #   print(features[i].shape)
    for i in features:
        i = i.detach()

    feat,seg,dep= R_D(features,syn_features1)

    for i in feat:
        print('feat',i.shape)
    print(seg.shape)
    print(dep[1].shape)
    for i in dep[0]:
        print(i.shape)
    FD  = _FeatureDiscriminator(input_nc=512,size=(72,24)).cuda()
    FD2 = _Discriminator(input_nc=256).cuda()
    FD3 = _Discriminator(input_nc=128).cuda()

    pred1 = FD(feat[0])
    pred2 = FD2(feat[1])
    pred3 = FD3(feat[2])
    print(pred2.shape)
    print(pred3.shape)
