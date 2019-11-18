import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet34
from functools import partial
import math


nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.01):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            # nn.GroupNorm(num_groups=32,num_channels=dim_out),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            # nn.GroupNorm(num_groups=32,num_channels=dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            # nn.GroupNorm(num_groups=32, num_channels=dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            # nn.GroupNorm(num_groups=32, num_channels=dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.GroupNorm(num_groups=32,num_channels=dim_out)
        self.branch5_bn = nn.BatchNorm2d(dim_out,momentum=bn_mom)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            # nn.GroupNorm(num_groups=32, num_channels=dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        global_feature = nonlinearity(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        #		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result

# class SPPblock(nn.Module):
#     def __init__(self, in_channels):
#         super(SPPblock, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
#
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
#         self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
#         self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
#         self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
#         self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
#
#         out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
#
#         return out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Attention(nn.Module):

    def __init__(self,channels,s = 16):

        super(Attention,self).__init__()

        #alpha
        self.s_conv = nn.Conv2d(channels,1,1)
        #beta
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels,channels // s,1,1,0,bias=False)
        self.fc2 = nn.Conv2d(channels // s,channels,1,1,0,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        net = self.s_conv(x)
        net = self.sigmoid(net)
        net = net * x
        net = self.avgpool(net)
        net = self.fc1(net)
        net = nonlinearity(net)
        net = self.fc2(net)
        net = self.sigmoid(net)
        net = net * x
        net = net + x
        return nonlinearity(net)

class Bottlenck(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(Bottlenck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=False))

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        x = nonlinearity(out + out1)
        return x


class Res_block(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(Res_block,self).__init__()
        self.inter = out_channel // 4
        self.block1 = Bottlenck(in_channel,self.inter)
        self.block2 = Bottlenck(self.inter,self.inter)
        self.block3 = Bottlenck(self.inter,self.inter)
        self.block4 = Bottlenck(self.inter,self.inter)
        self.conv1x1 = nn.Conv2d(in_channel,out_channel,1,bias=False)
        self.conv1x1_2 = nn.Conv2d(in_channel,self.inter,kernel_size=1,bias=False)

    def forward(self, x):
        res = self.conv1x1(x)
        xx = self.conv1x1_2(x)
        x1 = self.block1(x)
        x2 = self.block2(xx + x1)
        x3 = self.block3(xx + x1 + x2)
        x4 = self.block4(xx + x1 + x2 + x3)
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = nonlinearity(x + res)
        x = channel_shuffle(x,groups=4)

        return x


class Up(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(Up,self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1,bias=False)
        self.block = Res_block(out_channels, out_channels)
        self.attention = Attention(out_channels)

    def forward(self, x,x2):

        x = self.deconv(x) + x2
        x = self.block(x)
        x = self.attention(x)
        return x

def get_num_cls(channels,num_cls,upscale_factor = None,is_PixelShuffle = False):
    if is_PixelShuffle:
        assert upscale_factor is not None
        return nn.Sequential(
                nn.PixelShuffle(upscale_factor=upscale_factor),
                nn.Conv2d(channels // (upscale_factor * upscale_factor), channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Conv2d(channels, num_cls, 1)
            )
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(True),
        nn.Conv2d(channels, num_cls, 1)
    )


class Decoder(nn.Module):

    def __init__(self,num_cls,channels,size = (256,256),is_PixelShuffle = False):
        super(Decoder,self).__init__()
        self.size = size
        self.is_PixelShuffle = is_PixelShuffle
        self.dac_block = DACblock(channels[0])
        self.aspp = ASPP(channels[0],channels[0])
        self.cls2 = get_num_cls(channels[0],num_cls,upscale_factor=16,is_PixelShuffle=is_PixelShuffle) if is_PixelShuffle else get_num_cls(channels[0],num_cls)
        self.up1 = Up(channels[0],channels[1])
        self.cls3 = get_num_cls(channels[1],num_cls,upscale_factor=8,is_PixelShuffle=is_PixelShuffle) if is_PixelShuffle else get_num_cls(channels[1],num_cls)
        self.up2 = Up(channels[1],channels[2])
        self.cls4 = get_num_cls(channels[2],num_cls,upscale_factor=4,is_PixelShuffle=is_PixelShuffle) if is_PixelShuffle else get_num_cls(channels[2],num_cls)
        self.up3 = Up(channels[2],channels[3])
        self.cls5 = get_num_cls(channels[3],num_cls,upscale_factor=2,is_PixelShuffle=is_PixelShuffle) if is_PixelShuffle else get_num_cls(channels[3],num_cls)
        self.finnal = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[3], 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(channels[3], channels[3], 3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(True),
            # nn.Dropout(0.2, False),
            nn.Conv2d(channels[3], num_cls, 1)
        )

    def forward(self, outs):
        x = self.dac_block(outs[0])
        x = self.aspp(x)
        cls2 = self.cls2(x)
        if self.is_PixelShuffle is False:
            cls2 = F.upsample(cls2, size=self.size, mode='bilinear')
        x = self.up1(x,outs[1])
        cls3 = self.cls3(x)
        if self.is_PixelShuffle is False:
            cls3 = F.upsample(cls3, size=self.size, mode='bilinear')
        x = self.up2(x,outs[2])
        cls4 = self.cls4(x)
        if self.is_PixelShuffle is False:
            cls4 = F.upsample(cls4, size=self.size, mode='bilinear')
        x = self.up3(x,outs[3])
        cls5 = self.cls5(x)
        if self.is_PixelShuffle is False:
            cls5 = F.upsample(cls5, size=self.size, mode='bilinear')
        x = self.finnal(x)

        return cls2,cls3,cls4,cls5,x






class Network(nn.Module):

    def __init__(self,num_cls,channels = [512,256,128,64],is_train = False,is_PixelShuffle = False,**kwargs):
        super(Network,self).__init__()
        self.encoder = resnet34(dilated = False,pretrained = is_train,**kwargs)
        self.decoder = Decoder(num_cls=num_cls,channels=channels,is_PixelShuffle = is_PixelShuffle)
    def forward(self, x):
        x = self.encoder(x)
        outs = x[::-1]

        outs = self.decoder(outs)
        return outs

    # def init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


def change(input):
    mer = np.zeros((256,256,len(input)),dtype = np.int)
    for i,j in enumerate(outs):
        mer[:,:,i] = np.array(torch.argmax(j,dim=1)[0])
    pred = np.zeros((256,256),dtype = np.int)
    for i in range(256):
        nums = []
        for j in range(256):
            for k in range(len(input)):
                nums.append(mer[i,j,k])
        pred[i,j] = max(nums, key=nums.count)
    print(pred)



if __name__ == '__main__':
    import cv2
    import time
    import numpy as np
    x = torch.randn(1,4,256,256)
    t0 = time.time()
    model = Network(16,is_PixelShuffle = True)
    outs = model(x)
    meger = np.zeros((256,256,len(outs)),dtype = np.int)
    for i,j in enumerate(outs):
        meger[:,:,i] = np.array(torch.argmax(j,dim=1)[0])
    pred = np.zeros((256,256),dtype = np.int)
    for i in range(256):
        for j in range(256):
            print(meger[i,j,:])
            out = np.argmax(np.bincount(meger[i,j,:]))
            pred[i,j] = out
    print(pred)
    change(outs)
    print(time.time() - t0)


    # torch.save(model.state_dict(),'./test.pth')
