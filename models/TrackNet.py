"""TrackNet architecture - Slayminton's TensorFlow-to-PyTorch implementation.

Uses the working pretrained weights from models/tracknet.pt.

NOTE: This implementation includes a BatchNorm transpose hack to match TensorFlow behavior.
The 3rd parameter to Conv() is image width (not output channels) and is used for BatchNorm
initialization. This is NOT a bug - it's required for weight compatibility.
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Conv layer with optional TensorFlow BatchNorm compatibility.

    When tf_bn_compat=True (V2 pretrained weights), applies the TF BN hack
    (transpose to apply BN over width dimension).
    When tf_bn_compat=False (V3 or clean PyTorch), applies standard BN over channels.
    """
    def __init__(self, ic, oc, bc, k=3, p=1, act=True, tf_bn_compat: bool = False):
        super(Conv, self).__init__()
        self.tf_bn_compat = tf_bn_compat
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(bc)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.conv(x))

        if self.tf_bn_compat:
            # The TensorFlow-to-PyTorch hack for V2 weights!
            # Rotates tensor from [Batch, Channels, Height, Width]
            # to [Batch, Width, Height, Channels]
            x = x.transpose(1, 3)
            x = self.bn(x)           # Applies BN over the Width dimension!
            x = x.transpose(1, 3)    # Rotates it back
        else:
            # Standard PyTorch BatchNorm over channels
            x = self.bn(x)

        return x


class TrackNet(nn.Module):
    def __init__(self, out_channels=3, tf_bn_compat: bool = False):
        """TrackNet V2 architecture.

        Args:
            out_channels: Number of output channels (typically 1 for shuttle detection).
            tf_bn_compat: If True, apply TensorFlow BatchNorm compatibility (for V2 pretrained weights).
                         If False, use standard PyTorch BatchNorm (for V3 or clean training).
        """
        super(TrackNet, self).__init__()

        # Encoder (Notice the 3rd parameter is the Image Width, not Channels!)
        self.conv2d_1 = Conv(9, 64, 512, tf_bn_compat=tf_bn_compat)
        self.conv2d_2 = Conv(64, 64, 512, tf_bn_compat=tf_bn_compat)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_3 = Conv(64, 128, 256, tf_bn_compat=tf_bn_compat)
        self.conv2d_4 = Conv(128, 128, 256, tf_bn_compat=tf_bn_compat)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_5 = Conv(128, 256, 128, tf_bn_compat=tf_bn_compat)
        self.conv2d_6 = Conv(256, 256, 128, tf_bn_compat=tf_bn_compat)
        self.conv2d_7 = Conv(256, 256, 128, tf_bn_compat=tf_bn_compat)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_8 = Conv(256, 512, 64, tf_bn_compat=tf_bn_compat)
        self.conv2d_9 = Conv(512, 512, 64, tf_bn_compat=tf_bn_compat)
        self.conv2d_10 = Conv(512, 512, 64, tf_bn_compat=tf_bn_compat)

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_11 = Conv(768, 256, 128, tf_bn_compat=tf_bn_compat)
        self.conv2d_12 = Conv(256, 256, 128, tf_bn_compat=tf_bn_compat)
        self.conv2d_13 = Conv(256, 256, 128, tf_bn_compat=tf_bn_compat)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_14 = Conv(384, 128, 256, tf_bn_compat=tf_bn_compat)
        self.conv2d_15 = Conv(128, 128, 256, tf_bn_compat=tf_bn_compat)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_16 = Conv(192, 64, 512, tf_bn_compat=tf_bn_compat)
        self.conv2d_17 = Conv(64, 64, 512, tf_bn_compat=tf_bn_compat)

        self.conv2d_18 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.conv2d_2(self.conv2d_1(x))
        x_pool1 = self.maxpool1(x1)

        x2 = self.conv2d_4(self.conv2d_3(x_pool1))
        x_pool2 = self.maxpool2(x2)

        x3 = self.conv2d_7(self.conv2d_6(self.conv2d_5(x_pool2)))
        x_pool3 = self.maxpool3(x3)

        x4 = self.conv2d_10(self.conv2d_9(self.conv2d_8(x_pool3)))

        # Decoder with Skip Connections
        up1 = self.upsample1(x4)
        concat1 = torch.cat([up1, x3], dim=1)
        d1 = self.conv2d_13(self.conv2d_12(self.conv2d_11(concat1)))

        up2 = self.upsample2(d1)
        concat2 = torch.cat([up2, x2], dim=1)
        d2 = self.conv2d_15(self.conv2d_14(concat2))

        up3 = self.upsample3(d2)
        concat3 = torch.cat([up3, x1], dim=1)
        d3 = self.conv2d_17(self.conv2d_16(concat3))

        out = self.sigmoid(self.conv2d_18(d3))
        return out
