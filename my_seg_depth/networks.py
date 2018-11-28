import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch-10) / float(30)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

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

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)
    net.to(gpu_ids[0])
    init_weights(net, init_type, gain=init_gain)
    return net

class G_1(nn.Module):
    def __init__(self, input_nc, out_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,n_blocks=6,padding_type = 'reflect'):
        assert(n_blocks>0)
        super(G_1, self).__init__()
        self.input_nc = input_nc
        self.out_nc = out_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model =  [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc,ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
           # print('----',i)
            model += [nn.Conv2d(ngf*mult, ngf*mult*2,kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf*mult*2),nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks):
           # print('++++',i)
            model += [ResnetBlock(ngf*mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,use_bias=use_bias)]
      #  for i in range(n_downsampling):
          #  print('....',i)
      #      mult = 2**(n_downsampling-i)
      #      model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult / 2),
      #                                   kernel_size=3,stride=2,
      #                                   padding=1,output_padding=1,
      #                                   bias=use_bias),
      #                norm_layer(int(ngf*mult/2)),
      #                nn.ReLU(True)
      #                ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf*mult,out_nc, kernel_size=7,padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)

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
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class Feature_net(nn.Module):
    def __init__(self,input_nc,out_nc,growth_rate=32, block_config=(6, 12, 12, 12),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(Feature_net, self).__init__()
        self.features = []

        # Each denseblock

        num_features = input_nc
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('centerblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('padding5',nn.ReflectionPad2d(3))
        self.features.add_module('conv5',nn.Conv2d(num_features, out_nc, kernel_size=7, padding=0))
        self.features.add_module('relu5',nn.ReLU())


        def forward()


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





if __name__ == '__main__':
    x = torch.Tensor(1, 3, 256, 256).cuda()
    G = G_1(input_nc=3,out_nc=128).cuda()
    G2 = G_1(input_nc=3,out_nc=128).cuda()
    y = G(x)
    print(G._modules)
    print(y.shape)









