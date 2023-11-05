# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import torch
import torch.nn as nn
import math
import torch.nn.init


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss
class CML(nn.Module):

    def __init__(self,
                 in_channels=72,
                 out_channels=72,
                 groups=18,
                 width_per_group=4,
                 # bottleneck_width,
                 scales=4,
                 # base_width=64,

                 **kwargs):
        super(CML, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.width_per_group = width_per_group

        self.mid_channels = (
                groups * width_per_group)

        self.norm1 = nn.BatchNorm2d(
            self.mid_channels)
        self.norm3 = nn.BatchNorm2d(
            self.out_channels)

        self.conv1 = nn.Conv2d(

            self.in_channels,
            self.mid_channels,
            kernel_size=1,

            bias=False)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                nn.Conv2d(

                    self.mid_channels,
                    self.mid_channels,
                    groups=self.groups,
                    kernel_size=3,
                    padding=1,
                    bias=False))
            self.bns.append(
                nn.BatchNorm2d(self.mid_channels))

        self.conv3 = nn.Conv2d(

            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)

        self.scales = scales
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        spx = out
        sp = spx

        for i in range(self.scales - 1):
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp)) + spx

        out = sp
        out = self.conv3(out)
        out = self.norm3(out)
        out += identity
        out = self.relu(out)

        return out

class CMLNet(nn.Module):
    def __init__(self):
        super(CMLNet, self).__init__()

        channel = 72
        spectral_num = 8

        self.conv0 = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=6, stride=4,
                                        padding=1,
                                        bias=True)

        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1 , out_channels=channel, kernel_size=3, padding=1)

        self.res1 = CML()
        self.res2 = CML()
        self.res3 = CML()
        self.res4 = CML()

        self.res = nn.Sequential(
            self.res1,

            self.res2,

            self.res3,

            self.res4,


        )

        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        init_weights(self.res, self.conv0, self.conv1, self.conv2)

    def forward(self, x,y):  # x = ms; y = pan

        x = self.relu(self.conv0(x))
        z = torch.cat([x, y], 1)
        input = self.relu(self.conv1(z))

        rs1 = self.res(input)
        rs1 = self.relu(self.conv2(rs1))

        pred1 = torch.mul(x, rs1)
        return pred1


# ----------------- End-Main-Part ------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor



