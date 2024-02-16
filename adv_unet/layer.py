import torch
import torch.nn as nn

# For U-Net, define the double convolutional layer
class DoubleConv(nn.Module):
    '''
    This is the double convolutional structure
    我自己写的U-Net中，每一次down sampling或者up sampling之后所需要的两次卷积结构
    '''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), # one 2D convolution
            nn.InstanceNorm3d(out_channels), # one 2D Instance Norm to make the training more stable
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), # one 2D convolution
            nn.InstanceNorm3d(out_channels), # one 2D Instance Norm to make the training more stable
            nn.LeakyReLU(0.2, inplace=True) # one leaky relu for non-linear relationship
        )

    def forward(self, x):
        return self.conv(x)

class UnetBlock(nn.Module):
    '''
    This is the UnetBlock for U-NetGenerator
    Pix2pix的原代码 写的是一个recursion的U-Net，这个class是一个recursion的class
    '''
    def __init__(self, in_channels, out_channels, inner_channels,
                 submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv3d(in_channels=in_channels, out_channels=inner_channels, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.InstanceNorm3d(inner_channels)
        uprelu = nn.ReLU(inplace=True)
        upnorm = nn.InstanceNorm3d(out_channels)

        if outermost:
            upconv = nn.ConvTranspose3d(in_channels=inner_channels * 2, out_channels=out_channels,
                                        kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(in_channels=inner_channels, out_channels=out_channels,
                                        kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(in_channels=inner_channels * 2, out_channels=out_channels,
                                        kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)