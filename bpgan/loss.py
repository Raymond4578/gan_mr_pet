import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##################################################
# The following contents are all for KL Loss
##################################################

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar):
        output = torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * (- 0.5) # .pow(2)就是所有元素二次方
        return output

##################################################
# The following contents are all for GP Loss
##################################################

class GPLoss(nn.Module):
    '''
    I write this loss based on a 2D GP loss found from: https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/pytorch_spl_loss.py
    For details, please find the link above.
    '''
    def __init__(self):
        super(GPLoss, self).__init__()

    @staticmethod
    def get_gradient(img):
        z_dim = F.pad(img, [0, 1, 0, 0, 0, 0])[..., :, :, 1:]
        y_dim = F.pad(img, [0, 0, 0, 1, 0, 0])[..., :, 1:, :]
        x_dim = F.pad(img, [0, 0, 0, 0, 0, 1])[..., 1:, :, :]

        dx, dy, dz = x_dim - img, y_dim - img, z_dim - img

        dx[:, :, -1, :, :] = 0
        dy[:, :, :, -1, :] = 0
        dz[:, :, :, :, -1] = 0

        return dx, dy, dz

    @staticmethod
    def SPLoss(fake, true, dim='x'):
        if dim == 'x':
            dim = 2
        elif dim == 'y':
            dim = 3
        elif dim == 'z':
            dim = 4

        a = 0
        b = 0

        for i in range(fake.shape[0]):
            for j in range(fake.shape[1]):
                for k in range(fake.shape[dim]):
                    if dim == 2:
                        a += torch.trace(
                            torch.matmul(
                                F.normalize(fake[i, j, k, :, :], p=2, dim=1),
                                torch.t(F.normalize(true[i, j, k, :, :], p=2, dim=1))
                            )
                        ) / fake.shape[3]

                        b += torch.trace(
                            torch.matmul(
                                torch.t(F.normalize(fake[i, j, k, :, :], p=2, dim=0)),
                                F.normalize(true[i, j, k, :, :], p=2, dim=0)
                            )
                        ) / fake.shape[4]
                    elif dim == 3:
                        a += torch.trace(
                            torch.matmul(
                                F.normalize(fake[i, j, :, k, :], p=2, dim=1),
                                torch.t(F.normalize(true[i, j, :, k, :], p=2, dim=1))
                            )
                        ) / fake.shape[2]

                        b += torch.trace(
                            torch.matmul(
                                torch.t(F.normalize(fake[i, j, :, k, :], p=2, dim=0)),
                                F.normalize(true[i, j, :, k, :], p=2, dim=0)
                            )
                        ) / fake.shape[4]
                    elif dim == 4:
                        a += torch.trace(
                            torch.matmul(
                                F.normalize(fake[i, j, :, :, k], p=2, dim=1),
                                torch.t(F.normalize(true[i, j, :, :, k], p=2, dim=1))
                            )
                        ) / fake.shape[2]

                        b += torch.trace(
                            torch.matmul(
                                torch.t(F.normalize(fake[i, j, :, :, k], p=2, dim=0)),
                                F.normalize(true[i, j, :, :, k], p=2, dim=0)
                            )
                        ) / fake.shape[3]

        a = - torch.sum(a) / (fake.shape[0] * fake.shape[1])
        b = - torch.sum(b) / (fake.shape[0] * fake.shape[1])
        return a + b


    def forward(self, fake, true):
        fake_x, fake_y, fake_z = GPLoss.get_gradient(fake)
        true_x, true_y, true_z = GPLoss.get_gradient(true)

        trace_x = GPLoss.SPLoss(fake_x, true_x, 'x')
        trace_y = GPLoss.SPLoss(fake_y, true_y, 'y')
        trace_z = GPLoss.SPLoss(fake_z, true_z, 'z')
        return (trace_x + trace_y + trace_z) / 3


if __name__ == '__main__':
    x = torch.randn([3, 1, 64, 64, 64])
    y = x + torch.randn(x.shape) * 1
    # y = torch.randn([3, 1, 64, 64, 64])
    gp_loss = GPLoss()

    gp_loss_value = gp_loss(x, y)
    print(gp_loss_value.item())
