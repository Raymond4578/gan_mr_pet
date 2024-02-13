import torch
import torch.nn as nn

# For U-Net, define the single convolutional layer
class Conv(nn.Module):
    '''
    It is a single convolution structure
    我自己写的
    '''
    def __init__(self, in_channels, out_channels, padding=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding), # one 3D convolution
            nn.InstanceNorm3d(out_channels), # one 2D Instance Norm to make the training more stable
            nn.LeakyReLU(0.2) # one leaky relu for non-linear relationship
        )

    def forward(self, x):
        return self.conv(x)

def up_sample_layer(in_channels, out_channels):
    '''
    return a transposed convolution layer
    BicycleGAN原代码
    '''
    upconv = [
        # nn.Upsample(scale_factor=2, mode='trilinear'),
        # nn.ReflectionPad3d(1),
        # nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=4, stride=2, padding=1)
    ]
    return upconv

# For U-Net, define the bottonneck layer
class T(nn.Module):
    '''
    This is the multiple convolution module, refer to the paper:
    BPGAN: Brain PET synthesis from MRI using generative adversarial network for multi-modal Alzheimer’s disease diagnosis
    我自己写的
    '''
    def __init__(self, in_channels=512):
        super(T, self).__init__()

        self.dimension_reduction = nn.Conv3d(in_channels, int(in_channels / 2), kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(int(in_channels / 2), int(in_channels / 2), kernel_size=1, stride=1, padding=0)
        self.dimension_increase = nn.Conv3d(int(in_channels / 2), in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_dimension_reduction = self.dimension_reduction(x)
        # print(x_dimension_reduction.shape)
        x1 = self.conv(x_dimension_reduction)
        # print(x1.shape)
        x3 = self.conv(self.conv(x_dimension_reduction))
        # print(x3.shape)
        x_dimension_increase = self.dimension_increase(x1 + x_dimension_reduction + x3)
        # print(x_dimension_increase.shape)
        return x + x_dimension_increase

class UnetBlock(nn.Module):
    '''
    This is the UnetBlock for MCU_z_input
    BicycleGAN原代码
    '''
    def __init__(self, in_channels, out_channels, inner_channels,
                 submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost

        downconv = [nn.ReflectionPad3d(1)]

        downconv += [
            nn.Conv3d(in_channels, inner_channels, kernel_size=4, stride=2, padding=0)
        ]
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.InstanceNorm3d(inner_channels)
        uprelu = nn.ReLU(inplace=True)
        upnorm = nn.InstanceNorm3d(out_channels)

        if outermost:
            upconv = up_sample_layer(inner_channels * 2, out_channels)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = up_sample_layer(inner_channels, out_channels)
            down = [downrelu] + downconv
            up = [uprelu] + upconv + [upnorm]
            self.t = T(in_channels=inner_channels)
        else:
            upconv = up_sample_layer(inner_channels * 2, out_channels)
            down = [downrelu] + downconv + [downnorm]
            up = [uprelu] + upconv + [upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x):
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1)
            return self.up(x2)
        elif self.innermost:
            x1 = self.down(x)
            x2 = self.t(x1)
            return torch.cat([self.up(x2), x], dim=1)
        else:
            x1 = self.down(x)
            x2 = self.submodule(x1)
            return torch.cat([self.up(x2), x], dim=1)

class UnetBlock_with_z(nn.Module):
    '''
    This is the UnetBlock for MCU_z_all
    BicycleGAN原代码
    '''
    def __init__(self, in_channels, out_channels, inner_channels,
                 nz=0, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetBlock_with_z, self).__init__()

        downconv = [nn.ReflectionPad3d(1)]

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = in_channels + nz
        downconv += [
            nn.Conv3d(input_nc, inner_channels, kernel_size=4, stride=2, padding=0)
        ]
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        uprelu = nn.ReLU(inplace=True)

        if outermost: #这层是最后的输出了
            upconv = up_sample_layer(inner_channels * 2, out_channels)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = up_sample_layer(inner_channels, out_channels)
            down = [downrelu] + downconv
            up = [uprelu] + upconv + [nn.InstanceNorm3d(out_channels)]
            self.t = T(in_channels=inner_channels)
        else:
            upconv = up_sample_layer(inner_channels * 2, out_channels)
            down = [downrelu] + downconv + [nn.InstanceNorm3d(inner_channels)]
            up = [uprelu] + upconv + [nn.InstanceNorm3d(out_channels)]
            if use_dropout:
                up += [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(
                z.size(0), z.size(1), 1, 1, 1
            ).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4)
            )
            x_and_z = torch.cat([x, z_img], dim=1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.down(x_and_z)
            x2 = self.t(x1)
            return torch.cat([self.up(x2), x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1) # 这是个recursion！！！！好牛啊！！！！！什么天才写出来的啊！！！

# Residual Block
class ResBlock(nn.Module):
    '''
    This is the ResNet Block for ResNet Encoder
    BicycleGAN原代码
    '''
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        out_1 = self.conv(x)
        out_2 = self.shortcut(x)
        return out_1 + out_2


def test_conv():
    x = torch.randn((1, 1, 121, 145, 121))
    model = Conv(1, 64)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('Conv passed!')
    print()

def test_unet_block():
    x = torch.randn((1, 512, 2, 2, 2))
    z = torch.randn((1, 8))
    model = UnetBlock(in_channels=64 * 8, out_channels=64 * 8, inner_channels=64 * 8, innermost=True)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('U-Net Block passed!')
    print()

def test_unet_block_with_z():
    x = torch.randn((1, 512, 2, 2, 2))
    z = torch.randn((1, 8))
    model = UnetBlock_with_z(in_channels=64 * 8, out_channels=64 * 8, inner_channels=64 * 8, nz=8, innermost=True)
    preds = model(x, z)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('U-Net Block with z passed!')
    print()

def test_t():
    x = torch.randn((1, 512, 3, 4, 3))
    model = T(in_channels=512)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('T passed!')
    print()

def test_res():
    x = torch.randn((1, 64, 128, 128, 128))
    model = ResBlock(64, 128)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('Residual Block passed!')
    print()

if __name__ == '__main__':
    # test_conv()
    test_unet_block()
    # test_unet_block_with_z()
    # test_t()
    # test_res()