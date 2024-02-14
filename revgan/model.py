import torch
import torch.nn as nn

from layer import Encoder, Decoder, Reversible
from layer import RevBlock3d

class RevGAN(nn.Module):
    '''
    我自己写的RevGAN，encoder和decoder都是五层的CNN
    '''
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 512]):
        super(RevGAN, self).__init__()

        self.encoder_x = Encoder(in_channels, features)
        self.encoder_y = Encoder(out_channels, features)
        self.decoder_x = Decoder(features=list(reversed(features)), out_channels=in_channels)
        self.decoder_y = Decoder(features=list(reversed(features)), out_channels=out_channels)
        self.reversible_block = nn.ModuleList()

        for _ in range(4):
            self.reversible_block.append(Reversible(features[-1], features[-1]))

    def forward(self, x, inverse=False):
        if not inverse:
            x = self.encoder_x(x)
            for rev in self.reversible_block:
                x = rev(x)
            # print(x.shape)
            return self.decoder_y(x)
        else:
            # print('Enter reverse process!')
            y = self.encoder_y(x)
            for rev in nn.ModuleList(self.reversible_block[::-1]):
                y = rev(y, True)
            return self.decoder_x(y)

class RevGAN3d(nn.Module):
    '''
    3D RevGAN的原文
    '''
    def __init__(self, in_channels=1, out_channels=1, depth=4, ngf=64):
        super(RevGAN3d, self).__init__()

        downconv_ab = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels=in_channels, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            # 加入了2-2paper的新东西
            nn.InstanceNorm3d(ngf),
            nn.ReLU(inplace=True)
        ]
        downconv_ba = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels=in_channels, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            # 加入了2-2paper的新东西
            nn.InstanceNorm3d(ngf),
            nn.ReLU(inplace=True)
        ]

        core = []
        for _ in range(depth):
            core += [RevBlock3d(ngf)]
        upconv_ab = [
            nn.Conv3d(in_channels=ngf, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh() # 加入了2-2paper的新东西
        ]
        upconv_ba = [
            nn.Conv3d(in_channels=ngf, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()  # 加入了2-2paper的新东西
        ]

        self.downconv_ab = nn.Sequential(*downconv_ab)
        self.downconv_ba = nn.Sequential(*downconv_ba)
        self.core = nn.ModuleList(core)
        self.upconv_ab = nn.Sequential(*upconv_ab)
        self.upconv_ba = nn.Sequential(*upconv_ba)

    def forward(self, x, inverse=False):
        # orig_shape = x.shape[2:]
        out = x

        if not inverse:
            out = self.downconv_ab(out)
            for block in self.core:
                out = block(out)
            out = self.upconv_ab(out)
        else:
            out = self.downconv_ba(out)
            for block in reversed(self.core):
                out = block.inverse(out)
            out = self.upconv_ba(out)
        return out

class PatchGAN(nn.Module):
    '''
    RevGAN的原文，和BicycleGAN还有Pix2pix是完全一样的
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

        self.final_conv = nn.Conv3d(ndf * nf_mult, out_channels=1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        # print(x.shape)
        x = self.middle_conv(x)
        # print(x.shape)
        return self.final_conv(x)

# class PatchGAN(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1]):
#         super(PatchGAN, self).__init__()
#         self.conv = nn.ModuleList()
#         self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.Sigmoid = nn.Sigmoid()
#
#         for feature in features:
#             self.conv.append(nn.Conv3d(in_channels, feature, kernel_size=4, stride=1, padding=0))
#             in_channels = feature
#
#     def forward(self, x):
#         count = 0
#         for layer in self.conv:
#             x = layer(x)
#             if count in (0, 1):
#                 x = self.pool(x) # 没有pooling会跑得非常慢
#             # print(x.shape)
#             count += 1
#         return self.Sigmoid(x)

def test_patch():
    x = torch.randn((1, 1, 121, 145, 121))
    model = RevGAN(in_channels=1, out_channels=1, features=[64, 128, 256, 512, 512])
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('RevGAN forward passed!')
    print()

    y = torch.randn((1, 1, 121, 145, 121))
    model = RevGAN(in_channels=1, out_channels=1, features=[64, 128, 256, 512, 512])
    preds = model(x, True)
    print('The input data has the following shape:')
    print(y.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('RevGAN backward passed!')
    print()

    x = torch.randn((1, 2, 121, 145, 121))
    model = PatchGAN(in_channels=2)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('PatchGAN passed!')
    print()

def test_revblock3d():
    x = torch.randn((1, 1, 32, 32, 32))
    y = torch.randn((1, 1, 32, 32, 32))
    model = RevGAN3d(in_channels=1, out_channels=1, depth=4, ngf=64)
    preds_fwd = model(x)
    preds_bwd = model(y, True)
    print('The input data has the following shape:')
    print(x.shape)
    print(y.shape)
    print('The output data has the following shape:')
    print(preds_fwd.shape)
    print(preds_bwd.shape)
    print('RevGAN3D passed!')
    print()

if __name__ == '__main__':
    # test_patch()
    test_revblock3d()