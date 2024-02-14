import copy
import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    This is the encoder structure
    我自己写的RevGAN中的encoder
    '''
    def __init__(self, in_channels=1, features=[64, 128, 256, 512, 512]):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, feature, kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = feature

        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(out_channels),
        #     nn.ReLU()
        # )

    def forward(self, x):
        for encoder in self.encoder:
            x = encoder(x)
            # print(x.shape)
            x = self.pool(x)
            # print(x.shape)
            # print()
        return x

class Decoder(nn.Module):
    '''
    This is the decoder structure
    我自己写的RevGAN中的decoder
    '''
    def __init__(self, out_channels=1, features=list(reversed([64, 128, 256, 512, 512]))):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()

        for i, feature in enumerate(features):
            if i == len(features) - 1:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=0, output_padding=0),
                        nn.Conv3d(feature, out_channels, kernel_size=1, stride=1, padding=0),
                        nn.Tanh()
                    )
                )
            elif i == 0:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=0, output_padding=0),
                        nn.Conv3d(feature, features[i + 1], kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm3d(features[i + 1]),
                        nn.ReLU(inplace=True)
                    )
                )
            elif i == 1:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=(0, 1, 0), output_padding=(0, 1, 0)),
                        nn.Conv3d(feature, features[i + 1], kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm3d(features[i + 1]),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.Conv3d(feature, features[i + 1], kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm3d(features[i + 1]),
                        nn.ReLU(inplace=True)
                    )
                )



        # self.decoder = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        #     nn.Tanh()
        # )

    def forward(self, x):
        for decoder in self.decoder:
            x = decoder(x)
            # print(x.shape)
        return x

class RevSub(nn.Module):
    '''
    This is the structure of the irreversible subnetwork in the reversible block
    我自己写的RevGAN中的irrversible subnetwork
    '''
    def __init__(self, in_channels=1, out_channels=1):
        super(RevSub, self).__init__()
        self.rev_sub = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.rev_sub(x)

class Reversible(nn.Module):
    '''
    This is the structure of the reversible block, with additive coulping
    我自己写的RevGAN中的reversible block
    '''
    def __init__(self, in_channels=2, out_channels=2):
        super(Reversible, self).__init__()
        self.R1 = RevSub(int(in_channels / 2), int(out_channels / 2))
        self.R2 = RevSub(int(in_channels / 2), int(out_channels / 2))

    def forward(self, x, reverse=False):
        if not reverse:
            x1, x2 = torch.chunk(x, 2, dim=1)

            y1 = x1 + self.R1(x2)
            y2 = x2 + self.R2(y1)

            return torch.cat([y1, y2], dim=1)
        else:
            # print('Enter reverse process!')
            y = x

            y1, y2 = torch.chunk(y, 2, dim=1)

            x2 = y2 - self.R2(y1)
            x1 = y1 - self.R1(x2)

            return torch.cat([x1, x2], dim=1)

class AdditiveBlock(nn.Module):
    '''
    RevGAN原文的additive coupling
    '''
    def __init__(self, Fm, Gm):
        super(AdditiveBlock, self).__init__()
        self.Fm = Fm
        self.Gm = Gm

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        y1 = x1 + self.Fm.forward(x2)
        y2 = x2 + self.Gm.forward(y1)

        out = torch.cat([y1, y2], dim=1)

        x.data.set_()
        del x

        return out

    def inverse(self, y):
        y1, y2 = torch.chunk(y, chunks=2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        x2 = y2 - self.Gm.forward(y1)
        x1 = y1 - self.Gm.forward(x2)

        x = torch.cat([x1, x2], dim=1)

        y.data.set_()
        del y

        return x

class ReversibleBlock(nn.Module):
    '''
    RevGAN原文的additive coupling
    '''
    def __init__(self, Fm, Gm):
        super(ReversibleBlock, self).__init__()
        self.rev_block = AdditiveBlock(Fm, Gm)

    def forward(self, x):
        return self.rev_block(x)

    def inverse(self, y):
        return self.rev_block.inverse(y)

class ZeroInit(nn.Conv3d):
    '''
    RevGAN原文的特殊的一个3D convolutional layer
    '''
    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

class RevBlock3d(nn.Module):
    '''
    RevGAN原文的3D reversible block
    '''
    def __init__(self, dim):
        super(RevBlock3d, self).__init__()
        self.R1 = self.build_conv_block(dim // 2)
        self.R2 = self.build_conv_block(dim // 2)
        self.rev_block = ReversibleBlock(self.R1, self.R2)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            ZeroInit(dim, dim, kernel_size=3, padding=0)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.rev_block(x)

    def inverse(self, x):
        return self.rev_block.inverse(x)

def test_encoder():
    x = torch.randn((1, 1, 121, 145, 121))
    model = Encoder(in_channels=1, features=[64, 128, 256, 512, 512])
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('Encoder passed!')
    print()

def test_decoder():
    x = torch.randn((1, 512, 3, 4, 3))
    model = Decoder(out_channels=1, features=list(reversed([64, 128, 256, 512, 512])))
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('Decoder passed!')
    print()

def test_revsub():
    x = torch.randn((1, 1, 3, 4, 3))
    model = RevSub(in_channels=1, out_channels=1)
    preds = model(x)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('RevSub passed!')
    print()

def test_reversible():
    x = torch.randn((1, 2, 3, 4, 3))
    model = Reversible(in_channels=2, out_channels=2)
    preds = model(x, True)
    print('The input data has the following shape:')
    print(x.shape)
    print('The output data has the following shape:')
    print(preds.shape)
    print('Reversible passed!')
    print()

def test_revblock3d():
    x = torch.randn((1, 64, 32, 32, 32))
    y = torch.randn((1, 64, 32, 32, 32))
    model = RevBlock3d(dim=64)
    preds_fwd = model(x)
    preds_bwd = model(y)
    print('The input data has the following shape:')
    print(x.shape)
    print(y.shape)
    print('The output data has the following shape:')
    print(preds_fwd.shape)
    print(preds_bwd.shape)
    print('Rev Block 3D passed!')
    print()

if __name__ == '__main__':
    # test_encoder()
    # test_decoder()
    # test_revsub()
    # test_reversible()
    test_revblock3d()