import torch.nn as nn
import torch
def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)

def conv3x3(in_planes, out_planes, stride=1,dilation=0, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,dilation=dilation,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1,dilation=0, bias=False):
    "1x1 convolution"
    if dilation==0:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,dilation=dilation,
                     padding=0, bias=bias)

def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes),
                             nn.ReLU6())
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes))

class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,dilation=1,
                            bias=False),
                    )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class PSPBlock(nn.Module):

    def __init__(self,in_planes,out_planes,n):
        super(PSPBlock, self).__init__()
        for i in range(n):
            setattr(self, '{}_{}'.format(i + 1, 'pspconv'),
                    nn.Sequential(nn.Conv2d(in_planes, out_planes//4, kernel_size=3, stride=1,dilation=2*i+1,padding=2*i+1,
                               bias=False),
                    batchnorm(out_planes//4),
                    nn.ReLU6(),
                    ))
        self.conv = nn.Sequential(nn.Conv2d(in_planes,out_planes,1,1),batchnorm(out_planes),nn.ReLU6())

    def forward(self, x):
        top = x
        s = []
        for i in range(4):
            #  top = self.maxpool(top)
            s.append(getattr(self, '{}_{}'.format(i + 1, 'pspconv'))(top))
      #      print(s[i].shape)

        out = torch.cat([s[0],s[1],s[2],s[3]],dim=1)+self.conv(x)


        return out


class ATBlock(nn.Module):
    def __init__(self,df_c,out_c):
        super(ATBlock, self).__init__()

        # self.attention_bs = nn.Sequential(nn.Conv2d(df_c, df_c, 3, 2, padding=1)
        #                                   , batchnorm(df_c)
        #                                   , nn.ReLU6()
        #                                   , nn.AdaptiveAvgPool2d(1))

        self.attention_bs2 = nn.Sequential(nn.Conv2d(2*df_c, 2*df_c, 3, 2, padding=1)
                                          , nn.BatchNorm2d(2*df_c)
                                          , nn.LeakyReLU(0.02)
                                          , nn.AdaptiveAvgPool2d(1))
        self.at_act = nn.Sigmoid()
        self.conv = conv1x1(2*df_c,out_c)
       # self.conv1_s = conv1x1(df_c,df_c)
      #  self.conv1_f = conv1x1(2*df_c, df_c)

    def forward(self, s_feature, d_featuers):
       # attention = self.at_act(self.attention_bs(s_feature))
        #attention_ = self.at_act(attention)

        # attention = self.up(attention)
        s_f = s_feature
      #  s_f = torch.mul(attention, s_f)
       # s_f = self.conv1_s(s_f)
        x = torch.cat([s_f,d_featuers],dim=1)
        at2 = self.at_act(self.attention_bs2(torch.cat([s_f,d_featuers],dim=1)))

        out = torch.mul(at2,x)
        out = self.conv(out)

        return out

