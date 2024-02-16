import time
import argparse
import torch
import numpy as np
import random
import itertools
import nibabel as nib
import matplotlib.pyplot as plt

from utils import get_data_dir, load_data, split_data, get_data_batch
from utils import normalize_img
from utils import get_ssim, get_mae, get_psnr, get_rmse
from model import RevGAN, PatchGAN
from load_model import show_2d_slice

# 这个文件是训练RevGAN自己写的代码模型的文件

# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=5441, help='Random seed.')
parser.add_argument('--alpha', type=int, default=100, help='Weight for L1 loss for generator.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
parser.add_argument('--g_lr', type=float, default=0.0002, help='Initial generator learning rate.')
parser.add_argument('--d_lr', type=float, default=0.00001, help='Initial discriminator learning rate.')
parser.add_argument('--loss', type=int, default=1, help='Choose loss function. 0: Pix2pix, 1: CycleGAN')
parser.add_argument('--device', type=int, default=0, help='Setting the device of GPU, choose 0 or 1')

args = parser.parse_args() # 调用parse_args()方法进行解析；解析成功之后即可使用。# 这个是使用argparse模块时的必备行，将参数进行关联
# args现在是一个叫Namespace的object

device = args.device
args.cuda = not args.no_cuda and torch.cuda.is_available() # 给args加了一个量cuda=True

# set random seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# get the affine matrix to get nii file
data_dir = get_data_dir()
filename = data_dir + '/AV1451_PET_within_180_days/wADNI_002_S_0413_2017_06_21_PET_AV1451_PVC01_SUVR.nii'
affine_matrix = nib.load(filename).affine

dataloader = load_data()
print(f'number of pairs in the dataset: {len(dataloader)}')
data_train, data_test = split_data(dataloader, 0.8)
data_train = get_data_batch(data_train, args.batch_size)
data_test = get_data_batch(data_test, args.batch_size)

## Initialize generator and discriminator
generator = RevGAN()
discriminator_forward = PatchGAN()
discriminator_backward = PatchGAN()

# Loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()

if args.cuda:
    print('cuda used.')
    print(f'Using device {device}.')
    generator.cuda(device)
    discriminator_forward.cuda(device)
    discriminator_backward.cuda(device)
    adversarial_loss.cuda(device)
    l1_loss.cuda(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(discriminator_forward.parameters(), discriminator_backward.parameters()), lr=args.d_lr, betas=(0.5, 0.999))
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[100], gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[100], gamma=0.1)

def train(epoch, data_train, data_test):
    for i, entry in enumerate(data_train):
        # 每次训练一个batch的数据
        t = time.time()
        PET_img = entry[0]
        MR_img = entry[1]

        # get ready input x and output y
        x = MR_img.clone().detach()
        y = PET_img.clone().detach()
        if args.cuda:
            x = x.cuda(device)
            y = y.cuda(device)

        # normalize the image data

        x = normalize_img(x)
        y = normalize_img(y)

        # ---------------
        # Train Generator
        # ---------------

        optimizer_G.zero_grad()

        ###############################
        # from x to y, get predicted y
        ###############################
        y_hat = generator(x)

        fake_data_y = torch.cat([x, y_hat], dim=1)
        real_data_y = torch.cat([x, y], dim=1)

        # get ground choose
        y_true = torch.ones_like(discriminator_forward(real_data_y))
        y_false = torch.zeros_like(discriminator_forward(fake_data_y))
        if args.cuda:
            y_true = y_true.cuda(device)
            y_false = y_false.cuda(device)

        # get loss
        g_loss_adv_forward = adversarial_loss(discriminator_forward(fake_data_y), y_true)
        if args.loss == 0:
            g_loss_l1_forward = l1_loss(y_hat, y)
        elif args.loss == 1:
            g_loss_l1_forward = l1_loss(generator(y_hat, True), x)

        ###############################
        # from y to x, get predicted x
        ###############################
        x_hat = generator(y, True)

        fake_data_x = torch.cat([y, x_hat], dim=1)
        real_data_x = torch.cat([y, x], dim=1)

        # get ground choose
        x_true = torch.ones_like(discriminator_backward(real_data_x))
        x_false = torch.zeros_like(discriminator_backward(fake_data_x))
        if args.cuda:
            x_true = x_true.cuda(device)
            x_false = x_false.cuda(device)

        # get loss
        g_loss_adv_backward = adversarial_loss(discriminator_backward(fake_data_x), x_true)
        if args.loss == 0:
            g_loss_l1_backward = l1_loss(x_hat, x)
        elif args.loss == 1:
            g_loss_l1_backward = l1_loss(generator(x_hat), y)

        # finally get total loss
        g_loss = g_loss_adv_forward + g_loss_adv_backward + args.alpha * (g_loss_l1_forward + g_loss_l1_backward)

        # 反向传播
        g_loss.backward()
        torch.nn.utils.clip_grad_value_(generator.parameters(), 1) # 梯度裁剪
        optimizer_G.step()

        # -------------------
        # Train Discriminator
        # -------------------

        # backward propagation for forward discriminator
        optimizer_D.zero_grad()

        # measure forward discriminator loss
        real_loss_forward = adversarial_loss(discriminator_forward(real_data_y), y_true)
        fake_loss_forward = adversarial_loss(discriminator_backward(fake_data_y.detach()), y_false)
        d_loss_forward = (real_loss_forward + fake_loss_forward) * 0.5

        # backward propagation for backward discriminator
        # measure forward discriminator loss
        real_loss_backward = adversarial_loss(discriminator_backward(real_data_x), x_true)
        fake_loss_backward = adversarial_loss(discriminator_backward(fake_data_x.detach()), x_false)
        d_loss_backward = (real_loss_backward + fake_loss_backward) * 0.5

        d_loss = d_loss_forward + d_loss_backward

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 计算SSIM的结果并比较一下
        # ---------------------

        ssim_forward = get_ssim(y, y_hat)
        ssim_backward = get_ssim(x, x_hat)
        psnr_forward = get_psnr(y, y_hat)
        psnr_backward = get_psnr(x, x_hat)
        mae_forward = get_mae(y, y_hat)
        mae_backward = get_mae(x, x_hat)
        rmse_forward = get_rmse(y, y_hat)
        rmse_backward = get_rmse(x, x_hat)

        # -------
        # 打印结果
        # -------

        print(
            f'Epoch: {epoch + 1:03d}/{args.epochs},',
            f'Batch: {i + 1:03d}/{len(data_train)},',
            f'G Loss: {g_loss.item():.4f},',
            f'D Loss: {d_loss.item():.4f},',
            f'RMSE: {np.average(rmse_forward):.4f}/{np.average(rmse_backward):.4f},',
            f'MAE: {np.average(mae_forward):.4f}/{np.average(mae_backward):.4f},',
            f'PSNR: {np.average(psnr_forward):.4f}/{np.average(psnr_backward):.4f},',
            f'SSIM: {np.average(ssim_forward):.4f}/{np.average(ssim_backward):.4f},',
            f'Time: {time.time() - t:.4f}s.'
        )

    test(data_test)

def test(data_test):
    generator.eval()

    rmse_fwd_ls = []
    rmse_bwd_ls = []
    mae_fwd_ls = []
    mae_bwd_ls = []
    psnr_fwd_ls = []
    psnr_bwd_ls = []
    ssim_fwd_ls = []
    ssim_bwd_ls = []

    for i, entry in enumerate(data_test):
        PET_img = entry[0]
        MR_img = entry[1]

        x = MR_img.clone().detach()
        y = PET_img.clone().detach()
        if args.cuda:
            x = x.cuda(device)
            y = y.to(f'cuda:{args.device}')

        x = normalize_img(x)
        y = normalize_img(y)

        with torch.no_grad():
            y_hat = generator(x)
            x_hat = generator(y, True)

        rmse_fwd_ls = rmse_fwd_ls + get_rmse(y, y_hat)
        rmse_bwd_ls = rmse_bwd_ls + get_rmse(x, x_hat)
        mae_fwd_ls = mae_fwd_ls + get_mae(y, y_hat)
        mae_bwd_ls = mae_bwd_ls + get_mae(x, x_hat)
        psnr_fwd_ls = psnr_fwd_ls + get_psnr(y, y_hat)
        psnr_bwd_ls = psnr_bwd_ls + get_psnr(x, x_hat)
        ssim_fwd_ls = ssim_fwd_ls + get_ssim(y, y_hat)
        ssim_bwd_ls = ssim_bwd_ls + get_ssim(x, x_hat)

    print(
        'Test set results:',
        f'RMSE = {np.average(rmse_fwd_ls):.4f}/{np.average(rmse_bwd_ls):.4f},',
        f'MAE = {np.average(mae_fwd_ls):.4f}/{np.average(mae_bwd_ls):.4f},',
        f'PSNR = {np.average(psnr_fwd_ls):.4f}/{np.average(psnr_bwd_ls):.4f},',
        f'SSIM = {np.average(ssim_fwd_ls):.4f}/{np.average(ssim_bwd_ls):.4f}.'
    )

    generator.train()
    return generator

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train(epoch, data_train, data_test)
    print(f'This epoch takes {(time.time() - epoch_start_time) / 60:.2f}min. ({time.time() - epoch_start_time:.2f}s)')
    scheduler_G.step()
    scheduler_D.step()

test(data_test)

torch.save(generator, f'./model/RevNET_loss_{args.loss}_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt')

# 以下的部分就是输出一个作为例子看看结果的图片，可以没有，代码基本与load_model.py中一致的
true_x = data_test[2][1][0, :, :, :, :].unsqueeze(0).cuda(device)
true_y = data_test[2][0][0, :, :, :, :].unsqueeze(0).cuda(device)
true_x = normalize_img(true_x)
true_y = normalize_img(true_y)

with torch.no_grad():
    new_y = generator(true_x).cpu()
    new_x = generator(true_y, True).cpu()

show_2d_slice(true_x, true_y, new_y, new_x, 20, loss=args.loss)

plt.clf()

show_2d_slice(true_x, true_y, new_y, new_x, 60, loss=args.loss)

plt.clf()

show_2d_slice(true_x, true_y, new_y, new_x, 100, loss=args.loss)

# new_y_np = new_y.detach().squeeze().cpu().numpy()
# test_img = nib.Nifti1Image(new_y_np, affine=affine_matrix)
# nib.save(test_img, f'test_output_loss_{args.loss}_{time.strftime("%y%m%d%H%M")}.nii')