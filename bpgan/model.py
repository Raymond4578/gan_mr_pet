import torch
import torch.nn as nn

from layer import Conv, T
from layer import UnetBlock, UnetBlock_with_z, ResBlock

# 2.3.1 Generator

class MCU(nn.Module):
    '''
    下面这个MCU的Multiple Convolution U-Net是我自己写的，down sampling用的 convolution + polling,
    up-sampling 在transposed convolution之后加了double convoluiton
    '''
    def __init__(self, in_channels=1 + 8, out_channels=1, features=[64, 128, 256, 512, 512]):
        super(MCU, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottom = T(features[-1])
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # dofine encoder
        for i, feature in enumerate(features):
            self.encoder.append(Conv(in_channels, feature))
            in_channels = feature

        # define decoder
        # count = 0
        for count, feature in enumerate(reversed(features)):
            if count == 0 or count == 1:
                self.decoder.append(nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, output_padding=1))
            else:
                self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, output_padding=1))

            self.decoder.append(Conv(feature * 2, feature))

        # define the final layer, this is just a mapping to output domain with kernel size 1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv3d(features[0], out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh() # We use relu for normalized image, [-1, 1]
            # nn.ReLU(inplace=True) # we use ReLU because the pixel value is positive
        )

    def forward(self, x, z):
        # initialize a list to save the matrix
        encoder_outputs = []
        # combine x and z
        z_img = z.view(
            z.size(0), z.size(1), 1, 1, 1
        ).expand(
            z.size(0), z.size(1), x.size(2), x.size(3), x.size(4)
        )
        x = torch.cat([x, z_img], dim=1)

        num = 1
        for encoder in self.encoder:
            # for each step of down-sampling
            # print(f'Convolutional Layer: {num}')
            x = encoder(x) # go through a down-sampling step
            encoder_outputs.append(x) # save the output matrix
            # print(x.shape)
            x = self.pool(x) # do max pooling
            # print(x.shape)
            num += 1

        # after dowm-sampling, go through bottomneck layer
        x = self.bottom(x)
        # print(f'Convolutional Layer: {num}')
        # print(x.shape)

        # Reverse the list for saved matrix, because it is pairwise by depth
        num = 1
        encoder_outputs = encoder_outputs[::-1]
        for idx in range(0, len(self.decoder), 2):
            # first go through deconvolution to double the image size
            x = self.decoder[idx](x)
            # print(f'Deconvolutional Layer: {num}')
            # print(x.shape)
            # extract the corresponding matrix in down-sampling
            previous_output = encoder_outputs[idx // 2]
            # print(previous_output.shape)
            # concat the two matrix
            concat = torch.cat((previous_output, x), dim=1)
            # then do double convolution for the concated matrix
            x = self.decoder[idx + 1](concat)

            num += 1

        return self.final_conv(x)  # go through the final mapping layer


class MCU_z_input(nn.Module):
    '''
    下面这个MCU的Multiple Convolution U-Net是基于BicycleGAN原代码的，
    z的添加方法是只在input的时候加的MCU
    '''
    def __init__(self, in_channels=1, out_channels=1, nz=8, num_downs=7, ngf=64, use_dropout=False):
        super(MCU_z_input, self).__init__()
        self.nz = nz
        unet_block = UnetBlock(in_channels=ngf * 8, out_channels=ngf * 8, inner_channels=ngf * 8,
                               submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetBlock(in_channels=ngf * 8, out_channels=ngf * 8, inner_channels=ngf * 8,
                                   submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(in_channels=ngf * 4, out_channels=ngf * 4, inner_channels=ngf * 8,
                               submodule=unet_block)
        unet_block = UnetBlock(in_channels=ngf * 2, out_channels=ngf * 2, inner_channels=ngf * 4,
                               submodule=unet_block)
        unet_block = UnetBlock(in_channels=ngf, out_channels=ngf, inner_channels=ngf * 2,
                               submodule=unet_block)
        unet_block = UnetBlock(in_channels=in_channels + nz, out_channels=out_channels, inner_channels=ngf,
                               submodule=unet_block, outermost=True)
        self.model = unet_block

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(
                z.size(0), z.size(1), 1, 1, 1
            ).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4)
            )
            x_with_z = torch.cat([x, z_img], dim=1)
        else:
            x_with_z = x

        return self.model(x_with_z)

class MCU_z_all(nn.Module):
    '''
    下面这个MCU的Multiple Convolution U-Net是基于BicycleGAN原代码的，
    z的添加方法是在每一次down sampling的时候都加z的MCU
    '''
    def __init__(self, in_channels=1, out_channels=1, nz=8, num_downs=7, ngf=64, use_dropout=False):
        super(MCU_z_all, self).__init__()
        self.nz = nz
        unet_block = UnetBlock_with_z(in_channels=ngf * 8, out_channels=ngf * 8, inner_channels=ngf * 8,
                                      nz=nz, submodule=None, innermost=True)
        unet_block = UnetBlock_with_z(in_channels=ngf * 8, out_channels=ngf * 8, inner_channels=ngf * 8,
                                      nz=nz, submodule=unet_block, use_dropout=use_dropout)
        for _ in range(num_downs - 6):
            unet_block = UnetBlock_with_z(in_channels=ngf * 8, out_channels=ngf * 8, inner_channels=ngf * 8,
                                          nz=nz, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock_with_z(in_channels=ngf * 4, out_channels=ngf * 4, inner_channels=ngf * 8,
                                      nz=nz, submodule=unet_block)
        unet_block = UnetBlock_with_z(in_channels=ngf * 2, out_channels=ngf * 2, inner_channels=ngf * 4,
                                      nz=nz, submodule=unet_block)
        unet_block = UnetBlock_with_z(in_channels=ngf, out_channels=ngf, inner_channels=ngf * 2,
                                      nz=nz, submodule=unet_block)
        unet_block = UnetBlock_with_z(in_channels=in_channels, out_channels=out_channels, inner_channels=ngf,
                                      nz=nz, submodule=unet_block, outermost=True) # 最上面一层
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


# 2.3.2 Discriminator
class PatchGAN(nn.Module):
    '''
    这个PatchGAN是完全基于BicycleGAN原文写的PatchGAN
    '''
    def __init__(self, in_channels=2, ndf=64, n_layers=3):
        super(PatchGAN, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        nf_mult = 1
        middle_conv = []
        for n in range(1, n_layers): # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            middle_conv += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        middle_conv += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.middle_conv = nn.Sequential(*middle_conv)

        self.final_conv = nn.Conv3d(ndf * nf_mult, out_channels=1, kernel_size=4, stride=1, padding=1) # map back to one channel

    def forward(self, x):
        x = self.first_conv(x)
        # print(x.shape)
        x = self.middle_conv(x)
        # print(x.shape)
        return self.final_conv(x)


# class PatchGAN(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, features=[64, 256, 512]):
#         super(PatchGAN, self).__init__()
#         self.conv = nn.ModuleList()
#         # self.pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
#         for i, feature in enumerate(features):
#             self.conv.append(
#                 nn.Sequential(
#                     nn.Conv3d(in_channels, feature, kernel_size=4, stride=2, padding=1),
#                     nn.InstanceNorm3d(feature),
#                     nn.LeakyReLU(0.2, inplace=True)
#                 )
#             )
#             in_channels = feature
#         self.final = nn.Conv3d(features[-1], out_channels, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         for conv in self.conv:
#             x = conv(x)
#             # print(x.shape)
#         return self.final(x)

# 2.3.3 Encoder
class ResNet(nn.Module):
    '''
    这个Encoder是一个完全基于BicycleGAN原代码的ResNet，注意这个输出是一个对于高斯分布的mean和variance的点估计
    '''
    def __init__(self, in_channels=1, out_channels=8, ndf=64, n_blocks=4):
        super(ResNet, self).__init__()
        max_ndf = 4

        self.first_conv = nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True)

        middle_conv = []
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            middle_conv += [ResBlock(input_ndf, output_ndf)]
        middle_conv += [
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=8)
        ]
        self.middle_conv = nn.Sequential(*middle_conv)

        self.fc = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, out_channels)])


    def forward(self,x):
        x_conv = self.first_conv(x)
        x_conv = self.middle_conv(x_conv)
        # print(x_conv.shape)
        conv_flat = x_conv.view(x.size(0), -1)
        # print(conv_flat.shape)
        output = self.fc(conv_flat)
        outputVar = self.fcVar(conv_flat)
        return output, outputVar


def test_mcu():
    x = torch.randn((1, 1, 128, 128, 128))
    z = torch.randn((1, 8))
    model = MCU(in_channels=1 + 8, out_channels=1, features=[64, 128, 256, 512, 512])
    preds = model(x, z)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('MCU passed!')
    print()

def test_mcu_z_input():
    x = torch.randn((1, 1, 128, 128, 128))
    z = torch.randn((1, 8))
    model = MCU_z_input(in_channels=1, out_channels=1, nz=8, num_downs=6, use_dropout=True)
    preds = model(x, z)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('MCU passed!')
    print()

def test_mcu_z_all():
    x = torch.randn((1, 1, 128, 128, 128))
    z = torch.randn((1, 8))
    model = MCU_z_all(in_channels=1, out_channels=1, nz=8, num_downs=6, use_dropout=True)
    preds = model(x, z)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('MCU passed!')
    print()

def test_patch():
    x = torch.randn((7, 2, 64, 64, 64))
    model = PatchGAN(in_channels=2, ndf=64, n_layers=3)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('PatchGAN passed!')
    print()

def test_encoder():
    x = torch.randn((1, 1, 128, 128, 128))
    model = ResNet(in_channels=1, out_channels=8, ndf=64, n_blocks=4)
    preds, predsVar = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print(predsVar.shape)
    print('ResNet passed!')
    print()


if __name__ == '__main__':
    # test_mcu()
    test_mcu_z_input()
    # test_mcu_z_all()
    # test_patch()
    # test_encoder()