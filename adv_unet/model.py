import torch
import torch.nn as nn

from layer import DoubleConv
from layer import UnetBlock

class UNet(nn.Module):
    '''
    这个U-Net是我自己写的，down sampling用的 double convolution + polling,
    up-sampling 在transposed convolution之后加了double convoluiton
    '''
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList() # encoding, down-sampling
        self.decoder = nn.ModuleList() # decoding, up-sampling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) # max pooling

        # define encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # define the bottomneck layer
        # add Instance Norm and leakyReLU for stable training
        self.middle = nn.Sequential(
            DoubleConv(features[-1], features[-1]),
            nn.InstanceNorm3d(features[-1]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # define decoder
        count = 0
        for feature in reversed(features):
            # manually set up the deconvolutional layer because our image size is problematic
            if count == 0:
                self.decoder.append(nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=0,
                                                       output_padding=0))
            elif count == 1:
                self.decoder.append(nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=(0, 1, 0),
                                                       output_padding=(0, 1, 0)))
            elif count == 2 or count == 3:
                self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1,
                                                       output_padding=1))
            else:
                self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=0,
                                                       output_padding=0))
            self.decoder.append(DoubleConv(feature * 2, feature))
            count += 1

        # define the final layer, this is just a mapping to output domain with kernel size 1 convolution
        self.final_conv = nn.Sequential(
            nn.Conv3d(features[0], out_channels, kernel_size=1, stride=1, padding=0),
            # nn.InstanceNorm3d(out_channels),
            # nn.ReLU(inplace=True)  # we use ReLU because the pixel value is positive
            nn.Tanh()
        )

    def forward(self, x):
        # initialize a list to save the matrix
        encoder_outputs = []

        for encoder in self.encoder:
            # for each step of down-sampling
            x = encoder(x) # go through a down-sampling step
            encoder_outputs.append(x) # save the output matrix
            x = self.pool(x) # do max pooling

        # after dowm-sampling, go through bottomneck layer
        x = self.middle(x)

        # Reverse the list for saved matrix, because it is pairwise by depth
        encoder_outputs = encoder_outputs[::-1]
        for idx in range(0, len(self.decoder), 2):
            # first go through deconvolution to double the image size
            x = self.decoder[idx](x)
            # extract the corresponding matrix in down-sampling
            previous_output = encoder_outputs[idx // 2]
            # concat the two matrix
            concat = torch.cat((previous_output, x), dim=1)
            # then do double convolution for the concated matrix
            x = self.decoder[idx + 1](concat)

        return self.final_conv(x) # go through the final mapping layer

class UnetGenerator(nn.Module):
    '''
    这个是Pix2pix原文的U-Net代码
    '''
    def __init__(self, in_channels=1, out_channels=1, num_downs=6, ngf=64, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        unet_block = UnetBlock(ngf * 8, ngf * 8, ngf * 8, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetBlock(ngf * 8, ngf * 8, ngf * 8, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, submodule=unet_block)
        self.model = UnetBlock(in_channels=in_channels, out_channels=out_channels, inner_channels=ngf,
                               submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

class PatchGAN(nn.Module):
    '''
    这个是我自己写的PatchGAN，是一个五层的CNN
    '''
    def __init__(self, in_channels=2, out_channels=1, features=[64, 64, 128, 256, 512, 1]):
        super(PatchGAN, self).__init__()
        self.conv = nn.ModuleList()
        self.LeakyReLU = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.conv.append(nn.Conv3d(in_channels, feature, kernel_size=5, stride=1, padding=2))
            in_channels = feature

    def forward(self, x):
        count = 0
        for Conv in self.conv: # go through all the convolutional layer
            x = Conv(x)
            # If reach third layer, do LeakyReLU activation
            if count == 2:
                x = self.LeakyReLU(x)
                # If reach sixth layer, do Sigmoid activation
            elif count == 5:
                x = self.Sigmoid(x)

            # all the layer need pooling expect the sixth layer
            if count < 5:
                x = self.pool(x)

            count += 1

        return x

# class PatchGAN(nn.Module):
#     '''
#     这个是pix2pix原文的patchGAN
#     '''
#     def __init__(self, in_channels=2, ndf=64, n_layers=3):
#         super(PatchGAN, self).__init__()
#
#         self.first_conv = nn.Sequential(
#             nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#         nf_mult = 1
#         middle_conv = []
#         for n in range(1, n_layers): # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             middle_conv += [
#                 nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
#                 nn.InstanceNorm3d(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, inplace=True)
#             ]
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         middle_conv += [
#             nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
#             nn.InstanceNorm3d(ndf * nf_mult),
#             nn.LeakyReLU(0.2, inplace=True)
#         ]
#         self.middle_conv = nn.Sequential(*middle_conv)
#
#         self.final_conv = nn.Conv3d(ndf * nf_mult, out_channels=1, kernel_size=4, stride=1, padding=1)
#
#     def forward(self, x):
#         x = self.first_conv(x)
#         # print(x.shape)
#         x = self.middle_conv(x)
#         # print(x.shape)
#         return self.final_conv(x)


def test():
    x = torch.randn((1, 1, 121, 145, 121))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('UNet passed!')
    print()
    x = torch.randn((1, 2, 121, 145, 121))
    model = PatchGAN(in_channels=2, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print('PatchGAN passed!')

def test_unet_g():
    x = torch.randn((1, 1, 128, 128, 128))
    model = UnetGenerator(in_channels=1, out_channels=1, num_downs=6, ngf=64, use_dropout=False)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('UNet passed!')
    print()
def test_patchgan():
    x = torch.randn((1, 1, 128, 128, 128))
    y = torch.randn((1, 1, 128, 128, 128))
    data = torch.cat([x, y], dim=1)
    model = PatchGAN(in_channels=2, ndf=64, n_layers=3)
    preds = model(data)
    print('The input data has the following shape:')
    print(data.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('PatchGAN passed!')
    print()


if __name__ == '__main__':
    # test()
    test_unet_g()
    test_patchgan()