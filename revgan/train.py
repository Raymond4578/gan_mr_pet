import time
import argparse
import torch
import numpy as np
import random
import itertools
import nibabel as nib
import matplotlib.pyplot as plt

from utils import get_data_dir, load_data, split_data, get_data_batch
from utils import split_for_patches, combine_pathces
from utils import normalize_img
from utils import set_requires_grad
from utils import predict_through_image_window
from utils import get_ssim, get_mae, get_psnr, get_rmse
from model import RevGAN, PatchGAN
from model import RevGAN3d
from load_model import show_2d_slice

# 这个文件是训练RevGAN原代码模型的文件

# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=5441, help='Random seed.')
parser.add_argument('--alpha', type=int, default=100, help='Weight for L1 loss for generator.')
parser.add_argument('--epochs', type=int, default=12, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=40, help='Number of batch size.')
parser.add_argument('--g_lr', type=float, default=0.0002, help='Initial generator learning rate.')
parser.add_argument('--d_lr', type=float, default=0.000001, help='Initial discriminator learning rate.')
parser.add_argument('--loss', type=int, default=0, help='Choose loss function. 0: Pix2pix, 1: CycleGAN')
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
data_test = get_data_batch(data_test, args.batch_size * 1)

## Initialize generator and discriminator
# generator = RevGAN()
generator = RevGAN3d()
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
        # 现在读进去了一个batch的数据
        t = time.time()
        PET_img = entry[0]
        MR_img = entry[1]

        # get ready input x and output y
        x = MR_img.clone().detach()
        y = PET_img.clone().detach()

        # normalize the image data

        x = normalize_img(x)
        y = normalize_img(y)

        # 然后切割图像
        x_windows = split_for_patches(x, patch_size=(32, 32, 32), overlap=0.05)
        y_windows = split_for_patches(y, patch_size=(32, 32, 32), overlap=0.05)

        pairs = list(zip(x_windows, y_windows))  # 先配对起来
        random.shuffle(pairs)  # 然后把这些配对打乱
        x_windows, y_windows = zip(*pairs)  # 然后把这些配对分离
        x_windows = list(x_windows)  # 然后转回list
        y_windows = list(y_windows)  # 然后转回list

        g_loss_ls = []
        d_loss_ls = []
        for j in range(len(x_windows)):
            x_window, idx = x_windows[j]
            y_window, idx_y = y_windows[j]
            if args.cuda:
                x_window = x_window.to(f'cuda:{args.device}')
                y_window = y_window.to(f'cuda:{args.device}')

            #####################################
            # 先得到所有要用的结果，预测的y_hat和x_hat
            #####################################
            y_hat_window = generator(x_window)
            fake_data_forward = torch.cat([x_window, y_hat_window], dim=1)
            real_data_forward = torch.cat([x_window, y_window], dim=1)
            x_hat_window = generator(y_window, inverse=True)
            fake_data_backward = torch.cat([y_window, x_hat_window], dim=1)
            real_data_backward = torch.cat([y_window, x_window], dim=1)

            # adversarial loss baseline
            true_mat = torch.ones_like(discriminator_forward(real_data_forward))
            false_mat = torch.zeros_like(discriminator_forward(fake_data_forward))
            if args.cuda:
                true_mat = true_mat.to(f'cuda:{args.device}')
                false_mat = false_mat.to(f'cuda:{args.device}')

            ###############
            # 然后计算G的Loss
            ###############

            set_requires_grad([discriminator_forward, discriminator_backward], False)

            # 计算x到y的Loss
            g_loss_adv_forward = adversarial_loss(discriminator_forward(fake_data_forward), true_mat)
            if args.loss == 0:
                g_loss_l1_forward = l1_loss(y_hat_window, y_window)
            elif args.loss == 1:
                g_loss_l1_forward = l1_loss(generator(y_hat_window, True), x_window)
            g_loss_forward = g_loss_adv_forward + args.alpha * g_loss_l1_forward
            # 计算y到x的Loss
            g_loss_adv_backward = adversarial_loss(discriminator_backward(fake_data_backward), true_mat)
            if args.loss == 0:
                g_loss_l1_backward = l1_loss(x_hat_window, x_window)
            elif args.loss == 1:
                g_loss_l1_backward = l1_loss(generator(x_hat_window), y_window)
            g_loss_backward = g_loss_adv_backward + args.alpha * g_loss_l1_backward
            # get the total Loss for G
            g_loss = g_loss_forward + g_loss_backward

            g_loss_ls.append(g_loss.item())

            optimizer_G.zero_grad()

            g_loss.backward()
            optimizer_G.step()

            ####################
            # 然后计算D1和D2的Loss
            ####################

            set_requires_grad([discriminator_forward, discriminator_backward], True)
            # 计算x到y的Loss
            real_loss_forward = adversarial_loss(discriminator_forward(real_data_forward), true_mat)
            fake_loss_forward = adversarial_loss(discriminator_forward(fake_data_forward.detach()), false_mat)
            d_loss_forward = real_loss_forward + fake_loss_forward
            # 计算y到x的Loss
            real_loss_backward = adversarial_loss(discriminator_backward(real_data_backward), true_mat)
            fake_loss_backward = adversarial_loss(discriminator_backward(fake_data_backward.detach()), false_mat)
            d_loss_backward = real_loss_backward + fake_loss_backward
            # get the total Loss for D
            d_loss = (d_loss_forward + d_loss_backward) * 0.5

            d_loss_ls.append(d_loss.item())

            optimizer_D.zero_grad()

            d_loss.backward()
            optimizer_D.step()

        generator.eval()
        # 现在出了对于patch的循环，我们重新把之前切成的块用目前对于所有patch训练过的generator重新predict y来计算这个batch的SSIM等
        y_hat_ls = predict_through_image_window(windows=x_windows, generator=generator, direction='fwd', device=args.device)
        y_hat = combine_pathces(x, y_hat_ls)
        if args.cuda:
            y = y.to(f'cuda:{args.device}')
            y_hat = y_hat.to(f'cuda:{args.device}')
        x_hat_ls = predict_through_image_window(windows=y_windows, generator=generator, direction='bwd', device=args.device)
        x_hat = combine_pathces(y, x_hat_ls)
        if args.cuda:
            x = x.to(f'cuda:{args.device}')
            x_hat = x_hat.to(f'cuda:{args.device}')
        generator.train()

        # plt.subplot(2, 2, 1)
        # true_img = x.detach().cpu().numpy()[0][0][:, :, 60]
        # plt.imshow(true_img, cmap='gray')
        # plt.colorbar()
        # plt.title('True MR')
        #
        # plt.subplot(2, 2, 2)
        # sample_img = y_hat.detach().cpu().numpy()[0][0][:, :, 60]
        # plt.imshow(sample_img, cmap='gray')
        # plt.colorbar()
        # plt.title('Estinated PET')
        #
        # plt.subplot(2, 2, 3)
        # true_y_img = y.detach().cpu().numpy()[0][0][:, :, 60]
        # plt.imshow(true_y_img, cmap='gray')
        # plt.colorbar()
        # plt.title('True PET')
        # plt.show()
        #
        # plt.clf()


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
            f'G Loss: {np.average(g_loss_ls):.4f},',
            f'D Loss: {np.average(d_loss_ls):.4f},',
            f'RMSE: {np.average(rmse_forward):.4f}/{np.average(rmse_backward):.4f},',
            f'MAE: {np.average(mae_forward):.4f}/{np.average(mae_backward):.4f},',
            f'PSNR: {np.average(psnr_forward):.4f}/{np.average(psnr_backward):.4f},',
            f'SSIM: {np.average(ssim_forward):.4f}/{np.average(ssim_backward):.4f},',
            f'Time: {time.time() - t:.4f}s.'
        )
        # torch.cuda.empty_cache()

    test(data_test=data_test, generator=generator)

def test(data_test, generator):
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
            x = x.to(f'cuda:{args.device}')
            y = y.to(f'cuda:{args.device}')

        x = normalize_img(x)
        y = normalize_img(y)

        x_windows = split_for_patches(x, patch_size=(32, 32, 32), overlap=0.05)
        y_windows = split_for_patches(x, patch_size=(32, 32, 32), overlap=0.05)
        y_hat_ls = predict_through_image_window(windows=x_windows, generator=generator, direction='fwd',
                                                device=args.device)
        x_hat_ls = predict_through_image_window(windows=y_windows, generator=generator, direction='bwd',
                                                device=args.device)
        y_hat = combine_pathces(x, y_hat_ls)
        x_hat = combine_pathces(y, x_hat_ls)
        if args.cuda:
            y_hat = y_hat.to(f'cuda:{args.device}')
            x_hat = x_hat.to(f'cuda:{args.device}')

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

torch.save(generator, f'./model/RevGAN_loss_{args.loss}_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt')

# 以下的部分就是输出一个作为例子看看结果的图片，可以没有，代码基本与load_model.py中一致的
true_x = data_test[0][1][0, :, :, :, :].unsqueeze(0).cuda(device)
true_y = data_test[0][0][0, :, :, :, :].unsqueeze(0).cuda(device)
true_x = normalize_img(true_x)
true_y = normalize_img(true_y)

generator.eval()
new_x_windows = split_for_patches(true_x, patch_size=(32, 32, 32), overlap=0.05)
new_y_windows = split_for_patches(true_y, patch_size=(32, 32, 32), overlap=0.05)
new_y_hat_ls = predict_through_image_window(windows=new_x_windows, generator=generator, direction='fwd', device=args.device)
new_x_hat_ls = predict_through_image_window(windows=new_y_windows, generator=generator, direction='bwd', device=args.device)
new_y_hat = combine_pathces(true_x, new_y_hat_ls)
new_x_hat = combine_pathces(true_y, new_x_hat_ls)
new_y = new_y_hat.to('cpu')
new_x = new_x_hat.to('cpu')

show_2d_slice(true_x, true_y, new_y, new_x, 20, loss=args.loss)

plt.clf()

show_2d_slice(true_x, true_y, new_y, new_x, 60, loss=args.loss)

plt.clf()

show_2d_slice(true_x, true_y, new_y, new_x, 100, loss=args.loss)

# new_y_np = new_y.detach().squeeze().cpu().numpy()
# test_img = nib.Nifti1Image(new_y_np, affine=affine_matrix)
# nib.save(test_img, f'test_output_loss_{args.loss}_{time.strftime("%y%m%d%H%M")}.nii')