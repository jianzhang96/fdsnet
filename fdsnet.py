"""FDSNet: An Accurate Real-Time Surface Defect Segmentation Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcblock import ContextBlock2d
from .cbam import SELayer,SpatialGate

__all__ = ['FDSNet', 'get_fdsnet', 'get_fdsnet_mt_voc', 'get_fdsnet_sd_voc','get_fdsnet_phone_voc']


class FDSNet(nn.Module):
    def __init__(self, num_classes, aux=False, backbone=None, jpu=False, pretrained_base=False, **kwargs):
        super(FDSNet, self).__init__()
        self.aux = aux
        self.encoder = Encoder(32, 48, 64, (64, 96, 128), 128)
        self.feature_fusion = FeatureFusionModule(64, 64, 128)
        self.classifier = Classifer(128, num_classes)

        if self.aux:
            self.edge_auxlayer = nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # nn.Dropout(0.1),
                nn.Conv2d(128, 1, 1,padding=1)
            )
            self.semantic_auxlayer = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(512 * 4, num_classes - 1),  # ignore the background class
            )


        self.__setattr__('exclusive',[ 'encoder', 'feature_fusion', 'classifier',
                          'edge_auxlayer','semantic_auxlayer'] if self.aux else ['encoder', 'feature_fusion', 'classifier'])

    def forward(self, x):
        size = x.size()[2:]
        x8, x16,x32, x_en = self.encoder(x)
        x, x8_aux = self.feature_fusion(x8, x_en, x16)
        x = self.classifier(x,size)

        outputs = []
        outputs.append(x)
        if self.aux:
            auxout = self.edge_auxlayer(x8_aux)   # boundary auxiliary supervision
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            auxout = F.sigmoid(auxout)
            outputs.append(auxout)

            auxout_se = self.semantic_auxlayer(x32)  # semantic auxiliary supervision
            auxout_se = F.sigmoid(auxout_se)
            outputs.append(auxout_se)

        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _GroupConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, groupss=8, **kwargs):
        super(_GroupConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=groupss),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class MBlock(nn.Module):
    """LinearBottleneck used in MobileNetV2 and SElayer added in MobileNetV3"""
    def __init__(self, in_channels, out_channels, seLayer=None, t=6, stride=2, **kwargs):
        super(MBlock, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_se = seLayer
        assert self.use_se in ['selayer','spatial',None]
        if self.use_se =='selayer':
            self.atten = SELayer(out_channels)
        elif self.use_se == 'spatial':
            self.atten = SpatialGate()

    def forward(self, x):
        out = self.block(x)
        if self.use_se:
            out = self.atten(out)
        if self.use_shortcut:
            out = x + out
        return out

class GCU(nn.Module):
    """Global Context Upsampling module"""
    def __init__(self, in_ch1=128, in_ch2=128, in_ch3=128):
        super(GCU, self).__init__()
        self.gcblock1 = ContextBlock2d(in_ch1, in_ch1,pool='avg')
        self.group_conv1 = _GroupConv(in_ch1, in_ch1, 1, 2)
        self.group_conv2 = _GroupConv(in_ch2, in_ch2, 1, 2)
        self.gcblock3 = ContextBlock2d(in_ch3, in_ch3)

    def forward(self, x32, x16, x8):
        x32 = self.gcblock1(x32)
        x16_32 = F.interpolate(x32, x16.size()[2:], mode='bilinear', align_corners=True)
        x16_32 = self.group_conv1(x16_32)
        x16_fusion = x16 + x16_32

        x8_16 = F.interpolate(x16_fusion, x8.size()[2:], mode='bilinear', align_corners=True)
        x8_fusion = torch.mul(x8 , x8_16)
        x8gp = self.group_conv2(x8_fusion)
        x8gc = self.gcblock3(x8gp)
        return x8gc

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class Encoder(nn.Module):
    """Global feature extractor module"""

    def __init__(self, dw_channels1=32, dw_channels2=48,in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(2, 2, 2), **kwargs):
        super(Encoder, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, in_channels, 2)

        self.mblock1 = self._make_layer(MBlock, in_channels, block_channels[0], num_blocks[0], t, 2,[None,None])
        self.mblock2 = self._make_layer(MBlock, block_channels[0], block_channels[1], num_blocks[1], t, 2,[None,'selayer'])
        self.mblock3 = self._make_layer(MBlock, block_channels[1], block_channels[2], num_blocks[2], t, 1,[None,None])

        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1,atten=None):
        layers = []
        layers.append(block(inplanes, planes,atten[0], t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes,atten[i], t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x --> 1/8x
        x = self.conv(x)
        x = self.dsconv1(x)
        x8 = self.dsconv2(x)
        # stack the MoileNetV3 blocks
        x16 = self.mblock1(x8)
        x32 = self.mblock2(x16)
        x32_2 = self.mblock3(x32)
        out = self.ppm(x32_2)
        return x8,x16, x32_2, out


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, x8_in_ch=64, x16_in_ch=64, out_channels=128,  **kwargs):
        super(FeatureFusionModule, self).__init__()

        self.conv_8 = nn.Sequential(
            nn.Conv2d(x8_in_ch, out_channels, 1),
            nn.BatchNorm2d(out_channels),)

        self.conv_16 = nn.Sequential(
            nn.Conv2d(x16_in_ch, out_channels, 1),
            nn.BatchNorm2d(out_channels))

        self.gcu = GCU(out_channels, out_channels, out_channels)

    def forward(self, x8, x32, x16):
        # 1*1 conv
        x8 = self.conv_8(x8)
        x16 = self.conv_16(x16)

        out = self.gcu(x32, x16, x8)

        return out,x8


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        # self.group_conv1 = _GroupConv(dw_channels, dw_channels, stride, 2)
        self.conv1 = nn.Conv2d(dw_channels, dw_channels, stride, )

        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x,size):
        x = self.dsconv1(x)
        # x = self.group_conv1(x)
        x = self.conv1(x)
        x = self.conv(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def get_fdsnet(dataset='citys', backbone='', pretrained=False, root='/data/zhangj/.torch/models',
               pretrained_base=False,
               **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'mt_voc': 'mt_voc',
        'sd_voc': 'sd_voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
        'phone_voc': 'phone_voc',
    }
    from ..data.dataloader import datasets
    model = FDSNet(datasets[dataset].NUM_CLASS, **kwargs)  
    if pretrained:
        from .model_store import get_model_file
        device = torch.device('cuda:0')  # kwargs['local_rank']
        print('fdsnet__%s_best_model' % ( acronyms[dataset]))
        model.load_state_dict(
            torch.load(get_model_file('fdsnet__%s_best_model' % ( acronyms[dataset]), root=root),
                       map_location=device))  # map_location=device
    return model


def get_fdsnet_mt_voc(**kwargs):
    return get_fdsnet('mt_voc', **kwargs)

def get_fdsnet_phone_voc(**kwargs):
    return get_fdsnet('phone_voc', **kwargs)

def get_fdsnet_sd_voc(**kwargs):
    return get_fdsnet('sd_voc', **kwargs)


if __name__ == '__main__':
    pass