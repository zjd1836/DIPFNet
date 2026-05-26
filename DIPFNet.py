import os
import time
import math
import cv2
import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0,groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DWConvNobr(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConvNobr, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        dd=identity * a_w
        out = identity * a_w * a_h

        return out

class Conv1x1(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, 1)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlockDS(nn.Module):
    def __init__(self, inplanes, planes, transform=False):
        super(BasicBlockDS, self).__init__()
        self.transform = transform
        self.conv1 = DWConvNobr(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DWConvNobr(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1),
            nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.transform:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',}

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


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

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

    def forward(self, x):
        feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)
        return feature


def resnet34(pretrained=True):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model


class fusion_block(nn.Module):
    def __init__(self):
        super(fusion_block, self).__init__()
        self.link1 = nn.Sequential(
            BasicBlockDS(256, 256),
            Conv1x1(256, 128))
        self.link2 = nn.Sequential(
            BasicBlockDS(256, 256),
            Conv1x1(256, 128))
        self.link3 = nn.Sequential(
            BasicBlockDS(192, 192),
            Conv1x1(192, 64))
        self.link4 = nn.Sequential(
            BasicBlockDS(96,96),
            Conv1x1(96, 32),
            BasicBlockDS(32, 32),
            CoordAtt(32, 32),
            nn.Conv2d(32, 1, 1))

    def forward(self, x1, x2, x3, x4):
        x4 = F.interpolate(x4, x3.shape[2:], mode='bilinear', align_corners=True)
        x4 = self.link1(x4)  # 128*32*32

        x34 = torch.cat((x3, x4), 1)  # 256*32*32
        x34 = F.interpolate(x34, x2.shape[2:], mode='bilinear', align_corners=True)
        x34 = self.link2(x34)  # 128*64*64

        x234 = torch.cat((x2, x34), 1) # 192*128*128
        x234 = F.interpolate(x234, x1.shape[2:], mode='bilinear', align_corners=True)
        x234 = self.link3(x234) # 64*128*128

        x1234 = torch.cat((x1, x234), 1) # 128*128*128
        result = self.link4(x1234)  # 1*128*128

        return result



class decoder_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.de_block1 = BasicBlockDS(in_channels, in_channels)
        self.de_block2 = BasicBlockDS(in_channels, in_channels)
        self.de_block3 = nn.Sequential(
            BasicBlockDS(2 * in_channels, 2 * in_channels),
            Conv1x1(2 * in_channels, in_channels))

        self.de_block4 = nn.Sequential(
            Conv1x1(3 * in_channels, in_channels),
            BasicBlockDS(in_channels, in_channels),
            BasicBlockDS(in_channels, in_channels),
            Conv1x1(in_channels, out_channels),
            CoordAtt(out_channels, out_channels))

    def forward(self, x1, x2):

        f_diff = torch.abs(x1 - x2)
        f_add = x1 + x2
        f_conc = torch.cat((x1, x2), dim=1)
        f_diff = self.de_block1(f_diff)
        f_add = self.de_block2(f_add)
        f_conc = self.de_block3(f_conc)
        f = torch.cat((f_diff, f_add, f_conc), dim=1)
        result = self.de_block4(f)

        return result

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.block1 = decoder_block(512, 256)
        self.block2 = decoder_block(256, 128)
        self.block3 = decoder_block(128, 64)
        self.block4 = decoder_block(64, 32)
        self.block5 = fusion_block()
        self._init_weight()

    def forward(self, x1, x2):
        x1_1, x2_1, x3_1, x4_1 = x1[0], x1[1], x1[2], x1[3]  # 64*128*128 128*64*64 256*32*32 512*16*16
        x1_2, x2_2, x3_2, x4_2 = x2[0], x2[1], x2[2], x2[3]
        y4 = self.block1(x4_1, x4_2)  # 256*16*16
        y3 = self.block2(x3_1, x3_2) # 128*32*32
        y2 = self.block3(x2_1, x2_2) # 64*64*64
        y1 = self.block4(x1_1, x1_2) # 32*128*128
        output = self.block5(y1, y2, y3, y4) # 32*128*128
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DIPFNet(nn.Module):
    def __init__(self):
        super(DIPFNet, self).__init__()
        self.encoder = resnet34()
        self.decoder = Decoder()

    def forward(self, A, B):
        x_size = A.size()
        output1 = self.encoder(A)
        output2 = self.encoder(B)
        output = self.decoder(output1, output2)
        output = F.interpolate(output, x_size[2:], mode='bilinear', align_corners=True)
        return output


if __name__ == '__main__':
    test_data1 = torch.rand(1, 3, 512, 512).cuda()
    test_data2 = torch.rand(1, 3, 512, 512).cuda()
    model = DIPFNet()
    model = model.cuda()
    output = model(test_data1, test_data2)
    flops, params = profile(model, inputs=(test_data1, test_data2))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
