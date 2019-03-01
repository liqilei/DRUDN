import torch
import torch.nn as nn
import models.modules.blocks as B

class DRUDN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs, upscale_factor, norm_type=None,
                 act_type='prelu'):
        super(DRUDN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        if upscale_factor == 3:
            stride = 3
            padding = 2
            projection_filter = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8

        self.num_recurs = num_recurs

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = B.MeanShift(rgb_mean, rgb_std)
        self.conv_in = B.ConvBlock(in_channels, num_features, kernel_size=3, act_type=act_type, norm_type=None)


        self.up1_1 = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=norm_type, act_type=act_type)
        self.up1_2 = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=norm_type, act_type=act_type)
        self.up1_3 = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=norm_type, act_type=act_type)
        self.down1_1 = B.ConvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=norm_type, act_type=act_type)
        self.down1_2 = B.ConvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=norm_type, act_type=act_type)
        self.down1_3 = B.ConvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=norm_type, act_type=None)
        self.deconv1 = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                     padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv_feat = B.ConvBlock(num_features, num_features, kernel_size=3, act_type=act_type, norm_type=None)
        self.conv_out = B.ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=None)
        self.add_mean = B.MeanShift(rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)
        residual = self.conv_in(x)
        out = residual

        for i in range(self.num_recurs):
            out = self.down1_3(self.up1_3(self.down1_2(self.up1_2(self.down1_1(self.up1_1(out))))))
            out = torch.add(out, residual)

        out = self.conv_feat(out)
        out = torch.add(out, residual)
        out = self.conv_out(self.deconv1(out))
        out = self.add_mean(out)
        return out