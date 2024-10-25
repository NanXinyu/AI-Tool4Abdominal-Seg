import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv2d(in_planes, out_planes,stride=2):
    return nn.Conv2d(in_planes, 
                     out_planes, 
                     kernel_size=3, 
                     stride=stride, 
                     padding=1, 
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class DownSampling(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
       
        self.first_layer = nn.Sequential(
            conv2d(in_planes, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
       
        self.down_layer1 = nn.Sequential(
            conv2d(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.down_layer2 = nn.Sequential(
            conv2d(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.down_layer3 = nn.Sequential(
            conv2d(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.final_layer = nn.Conv2d(512, 64, kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        # print(x.shape)
        x = self.first_layer(x)
        x = self.down_layer1(x)
        x = self.down_layer2(x)
        x = self.down_layer3(x)
        x = self.final_layer(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_prob = F.softmax(attn_score, dim=-1)
        # print(attn_prob.shape, v.shape)
        attn_output = torch.matmul(attn_prob, v)

        return attn_output
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self,
                 args,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        
        self.down_layers = DownSampling(1)
        self.final_layer = nn.Conv3d(
            in_channels=block_inplanes[3] * block.expansion,
            out_channels=64,
            kernel_size=(3,3,3),
            stride=(1,1,1),
            padding=(1,1,1)
        )
        self.bn = nn.BatchNorm3d(64)
        
        self.avgpool = nn.AdaptiveAvgPool3d((32, 1, 1))
        self.avgpool2d = nn.AdaptiveAvgPool2d((32,1))
        self.cross_layer1 = CrossAttention(1024)
        self.cross_layer2 = CrossAttention(1024)
        self.mlp_head_s = nn.Linear(64*16, 256)
        self.mlp_head_e = nn.Linear(64*16, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.9))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    
    def forward(self, xs):
        # print(len(x))
        x = xs[0]
        f = xs[1]
        s = xs[2]
        # print(x.shape,f.shape,s.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_layer(x)
        x = self.relu(self.bn(x))

        x = self.avgpool(x).flatten(2) # [B, C, 32, 1, 8]
        
        f = self.down_layers(f)
        

        f = self.avgpool2d(f).flatten(2)

        s = self.down_layers(s)
       
        s = self.avgpool2d(s).flatten(2)
        
        # print(x.shape, s.shape, f.shape)
        start = x[:,:,:16].flatten(1) + \
            self.cross_layer1(x[:,:,:16].flatten(1),f[:,:,:16].flatten(1),x[:,:,:16].flatten(1)) + \
            self.cross_layer2(x[:,:,:16].flatten(1),s[:,:,:16].flatten(1),x[:,:,:16].flatten(1))
        
        
        end = x[:,:,16:].flatten(1) + \
            self.cross_layer1(x[:,:,16:].flatten(1),f[:,:,16:].flatten(1),x[:,:,16:].flatten(1)) + \
            self.cross_layer2(x[:,:,16:].flatten(1),s[:,:,16:].flatten(1),x[:,:,16:].flatten(1))

        start = self.mlp_head_s(start).reshape(x.shape[0],1,-1)
        end = self.mlp_head_e(end).reshape(x.shape[0],1,-1)

        return start, end


def generate_model(args, **kwargs):
    model_depth = args.netDepth
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(args, BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(args, BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(args, BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(args, Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(args, Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(args, Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(args, Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
