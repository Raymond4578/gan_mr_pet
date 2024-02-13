import time
import argparse
import torch
import numpy as np
import random
import nibabel as nib
import pytorch_ssim
import matplotlib.pyplot as plt

from utils import get_data_dir, load_data, split_data, get_data_batch
from utils import split_for_patches, combine_pathces
from utils import normalize_img
from utils import get_random_sample, encode_latent_code, expand_latent_code
from utils import set_requires_grad
from utils import predict_through_image_window
from utils import get_mae, get_ssim, get_rmse, get_psnr
from model import MCU, PatchGAN, ResNet
from model import MCU_z_input, MCU_z_all
from loss import KLLoss, GPLoss
from load_model import show_2d_slice

# torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=5441, help='Random seed.')
parser.add_argument('--lambda_1', type=float, default=10.0, help='Weight for L1 loss for generator 1.')
parser.add_argument('--lambda_2', type=float, default=0.5, help='Weight for L1 loss for generator 2.')
parser.add_argument('--lambda_kl', type=float, default=0.01, help='Weight for KL loss.')
parser.add_argument('--lambda_gp', type=float, default=0.02, help='Weight for GP loss.')
parser.add_argument('--lambda_ssim', type=float, default=1.0, help='Weight for SSIM loss.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--niter', type=int, default=100, help='Number of epochs before learning rate decay.')
parser.add_argument('--batch_size', type=int, default=3, help='Number of batch size.')
parser.add_argument('--g_lr', type=float, default=0.0002, help='Initial generator learning rate.')
parser.add_argument('--d_lr', type=float, default=0.000001, help='Initial discriminator learning rate.')
parser.add_argument('--nz', type=int, default=8, help='The length of latent code z.')
parser.add_argument('--use_dropout', type=bool, default=True, help='Use dropout in MCU.')
parser.add_argument('--input', type=str, default='pet', help='Setting the input to MR image or PET image.')
parser.add_argument('--device', type=int, default=0, help='Setting the device of GPU, choose 0 or 1')

args = parser.parse_args() # 调用parse_args()方法进行解析；解析成功之后即可使用。# 这个是使用argparse模块时的必备行，将参数进行关联
# args现在是一个叫Namespace的object

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

## Initialize generator, discriminator and encoder
# 如果需要换模型就这里手动comment换一下
# 主要是觉得上面args内容写太多也挺麻烦
generator = MCU(in_channels=1 + args.nz)
# generator = MCU_z_input(in_channels=1, out_channels=1, nz=args.nz, num_downs=7, use_dropout=args.use_dropout)
# generator = MCU_z_all(in_channels=1, out_channels=1, nz=args.nz, num_downs=7, use_dropout=args.use_dropout)
discriminator_1 = PatchGAN()
discriminator_2 = PatchGAN()
encoder = ResNet(out_channels=args.nz)

# Loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()
kl_loss = KLLoss()
gp_loss = GPLoss()
ssim_loss = pytorch_ssim.SSIM3D(window_size=11)

if args.cuda:
    print('cuda used.')
    print(f'Using device {args.device}.')
    generator.cuda(args.device)
    discriminator_1.cuda(args.device)
    discriminator_2.cuda(args.device)
    encoder.cuda(args.device)
    adversarial_loss.cuda(args.device)
    l1_loss.cuda(args.device)
    kl_loss.cuda(args.device)
    gp_loss.cuda(args.device)
    ssim_loss.cuda(args.device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
optimizer_D1 = torch.optim.Adam(discriminator_1.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(discriminator_2.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

def get_scheduler(optimizer, args):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.epochs - args.niter + 1)
        return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

scheduler_G = get_scheduler(optimizer_G, args)
scheduler_D1 = get_scheduler(optimizer_D1, args)
scheduler_D2 = get_scheduler(optimizer_D2, args)
scheduler_E = get_scheduler(optimizer_E, args)

# scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[10], gamma=0.1)
# scheduler_D1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D1, milestones=[10], gamma=0.1)
# scheduler_D2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D2, milestones=[10], gamma=0.1)
# scheduler_E = torch.optim.lr_scheduler.MultiStepLR(optimizer_E, milestones=[10], gamma=0.1)

def train(epoch, data_train, data_test):
    for i, entry in enumerate(data_train):
        # 现在读进去了一个batch的数据
        t = time.time()
        PET_img = entry[0]
        MR_img = entry[1]

        # get ready input x and output y
        if args.input == 'mr': # do MR to PET training
            x = MR_img.clone().detach()
            y = PET_img.clone().detach()
        elif args.input == 'pet': # do PET to MR training
            x = PET_img.clone().detach()
            y = MR_img.clone().detach()

        # normalize the image data

        x = normalize_img(x)
        y = normalize_img(y)


        # 然后把它切成2份
        windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)
        y_windows = split_for_patches(y, patch_size=(128, 128, 128), overlap=0.8)

        pairs = list(zip(windows, y_windows)) # 先配对起来
        random.shuffle(pairs) # 然后把这些配对打乱
        windows, y_windows = zip(*pairs) # 然后把这些配对分离
        windows = list(windows) # 然后转回list
        y_windows = list(y_windows) # 然后转回list

        g_e_loss_ls = []
        g_loss_only_ls = []
        d_loss_1_ls = []
        d_loss_2_ls = []
        for j in range(0, len(windows), 2):
            # 现在在每个batch中，我们一次读进两个patch，分成encoded的一组和random的一组

            # ---------------------------
            # Train Generator and Encoder
            # ---------------------------

            x_window_encoded, idx_encoded = windows[j]
            x_window_random, idx_random = windows[j + 1]
            y_window_encoded, idx_y_encoded = y_windows[j]
            y_window_random, idx_y_random = y_windows[j + 1]

            if args.cuda:
                x_window_encoded = x_window_encoded.cuda(args.device)
                x_window_random = x_window_random.to(f'cuda:{args.device}')
                y_window_encoded = y_window_encoded.cuda(args.device)
                y_window_random = y_window_random.cuda(args.device)

            # get encoded z, 也就是Q(z|y)，就是根据true y来生成的
            Qzy, mu, logvar = encode_latent_code(encoder=encoder, y_window=y_window_encoded, device=args.device)

            # get random z, 这个是那个N(z)，就是随机生成的
            Nz = get_random_sample(x_window_encoded.size(0), nz=args.nz).cuda(args.device)

            ####################
            # 先处理 encoded data
            ####################

            y_hat_encoded = generator(x_window_encoded, Qzy)

            ####################
            # 再处理 random data
            ####################

            y_hat_random = generator(x_window_encoded, Nz)

            # get z
            z, mu_2, logvar_2 = encode_latent_code(encoder=encoder, y_window=y_hat_random, device=args.device)

            fake_data_encoded = torch.cat([x_window_encoded, y_hat_encoded], dim=1)
            real_data_encoded = torch.cat([x_window_encoded, y_window_encoded], dim=1)
            fake_data_random = torch.cat([x_window_encoded, y_hat_random], dim=1)
            real_data_random = torch.cat([x_window_random, y_window_random], dim=1)

            # get ground choose for adversarial loss
            y_true = torch.ones_like(discriminator_1(real_data_encoded))
            y_false = torch.zeros_like(discriminator_1(fake_data_encoded))
            if args.cuda:
                y_true = y_true.to(f'cuda:{args.device}')
                y_false = y_false.to(f'cuda:{args.device}')

            ##################
            # 然后计算G和E的Loss
            ##################

            set_requires_grad([discriminator_1, discriminator_2], False)

            g_e_loss_adv_encoded = adversarial_loss(discriminator_1(fake_data_encoded), y_true)
            g_e_loss_adv_random = adversarial_loss(discriminator_2(fake_data_random), y_true)

            g_e_loss_kl_encoded = kl_loss(mu, logvar)

            g_e_loss_l1_encoded = l1_loss(y_hat_encoded, y_window_encoded)
            g_e_loss_l1_random = l1_loss(mu_2, Nz)

            g_e_loss = g_e_loss_adv_encoded + g_e_loss_adv_random + args.lambda_kl * g_e_loss_kl_encoded + args.lambda_1 * g_e_loss_l1_encoded
            g_loss_only = args.lambda_2 * g_e_loss_l1_random

            # 接下来是这个BPGAN比BicycleGAN多加出来的loss
            g_e_loss_gp_encoded = gp_loss(y_hat_encoded, y_window_encoded)

            g_e_loss_ssim_encoded = ssim_loss(y_hat_encoded, y_window_encoded)

            g_e_loss = g_e_loss + args.lambda_gp * g_e_loss_gp_encoded - args.lambda_ssim * g_e_loss_ssim_encoded

            # record the loss in this loop
            g_e_loss_ls.append(g_e_loss.item())
            g_loss_only_ls.append(g_loss_only.item())

            optimizer_G.zero_grad()
            optimizer_E.zero_grad()

            g_e_loss.backward(retain_graph=True)
            set_requires_grad([encoder], False)
            g_loss_only.backward()
            set_requires_grad([encoder], True)
            optimizer_G.step()
            optimizer_E.step()

            # --------------
            # Get Loss
            # --------------

            ############
            # 先处理D1
            ############

            set_requires_grad([discriminator_1], True)

            # measure forward discriminator loss
            real_loss_1 = adversarial_loss(discriminator_1(real_data_encoded), y_true)
            fake_loss_1 = adversarial_loss(discriminator_1(fake_data_encoded.detach()), y_false)
            d_loss_1 = (real_loss_1 + fake_loss_1) * 1

            d_loss_1_ls.append(d_loss_1.item())

            optimizer_D1.zero_grad()

            d_loss_1.backward()
            optimizer_D1.step()

            ############
            # 再处理D2
            ############

            set_requires_grad([discriminator_2], True)

            # measure forward discriminator loss
            real_loss_2 = adversarial_loss(discriminator_2(real_data_random), y_true)
            fake_loss_2 = adversarial_loss(discriminator_2(fake_data_random.detach()), y_false)
            d_loss_2 = (real_loss_2 + fake_loss_2) * 1

            d_loss_2_ls.append(d_loss_2.item())

            optimizer_D2.zero_grad()

            d_loss_2.backward()
            optimizer_D2.step()

        generator.eval()
        # 现在出了对于patch的循环，我们重新把之前切成的块用目前对于所有patch训练过的generator重新predict y来计算这个batch的SSIM等
        y_hat_ls = predict_through_image_window(windows=windows, generator=generator, nz=args.nz, device=args.device)
        y_hat = combine_pathces(x, y_hat_ls)
        if args.cuda:
            y = y.cuda(args.device)
            y_hat = y_hat.cuda(args.device)
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

        ssim_result = get_ssim(y, y_hat)
        psnr_result = get_psnr(y, y_hat)
        mae_result = get_mae(y, y_hat)
        rmse_result = get_rmse(y, y_hat)

        # -------
        # 打印结果
        # -------

        print(
            f'Epoch: {epoch + 1:03d}/{args.epochs},',
            f'Batch: {i + 1:03d}/{len(data_train)},',
            f'G&E Loss: {np.average(g_e_loss_ls):+.4f},',
            f'G Only Loss: {np.average(g_loss_only_ls):+.4f},',
            f'D1 Loss: {np.average(d_loss_1_ls):.4f},',
            f'D2 Loss: {np.average(d_loss_2_ls):.4f},',
            f'RMSE: {np.average(rmse_result):.4f},',
            f'MAE: {np.average(mae_result):.4f},',
            f'PSNR: {np.average(psnr_result):.4f},',
            f'SSIM: {np.average(ssim_result):.4f},',
            f'Time: {time.time() - t:.4f}s.'
        )

    test(data_test, generator, input_img=args.input)



def test(data_test, generator, input_img):
    generator.eval()

    rmse_ls = []
    mae_ls = []
    psnr_ls = []
    ssim_ls = []

    for i, entry in enumerate(data_test):
        PET_img = entry[0]
        MR_img = entry[1]

        # get ready input x and output y
        if input_img == 'mr': # do MR to PET training
            x = MR_img.clone().detach()
            y = PET_img.clone().detach()
        elif input_img == 'pet': # do PET to MR training
            x = PET_img.clone().detach()
            y = MR_img.clone().detach()
        if args.cuda:
            x = x.cuda(args.device)


        x = normalize_img(x)
        y = normalize_img(y)


        # 切割图像来得到小块，把它切成2份
        windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)

        y_hat_ls = predict_through_image_window(windows=windows, generator=generator, nz=args.nz, device=args.device)
        y_hat = combine_pathces(x, y_hat_ls)

        if args.cuda:
            y = y.cuda(args.device)
            y_hat = y_hat.cuda(args.device)

        rmse_ls = rmse_ls + get_rmse(y, y_hat)
        mae_ls = mae_ls + get_mae(y, y_hat)
        psnr_ls = psnr_ls + get_psnr(y, y_hat)
        ssim_ls = ssim_ls + get_ssim(y, y_hat)

    print(
        'Test set results:',
        f'RMSE = {np.average(rmse_ls):.4f},',
        f'MAE = {np.average(mae_ls):.4f},',
        f'PSNR = {np.average(psnr_ls):.4f},',
        f'SSIM = {np.average(ssim_ls):.4f}.'
    )

    generator.train()

    return generator

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train(epoch, data_train, data_test)
    print(f'This epoch takes {(time.time() - epoch_start_time) / 3600:.2f}h. ({time.time() - epoch_start_time:.2f}s)')
    scheduler_G.step()
    scheduler_E.step()
    scheduler_D1.step()
    scheduler_D2.step()

torch.save(generator, f'./model/BPGAN_input_{args.input}_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt')

# 以下的部分就是输出一个作为例子看看结果的图片，可以没有，代码基本与load_model.py中一致的
if args.input == 'mr':
    new_x = data_test[2][1][0, :, :, :, :].unsqueeze(0).cuda(args.device)
    true_y = data_test[2][0][0, :, :, :, :].unsqueeze(0)
elif args.input == 'pet':
    new_x = data_test[2][0][0, :, :, :, :].unsqueeze(0).cuda(args.device)
    true_y = data_test[2][1][0, :, :, :, :].unsqueeze(0)
new_x = normalize_img(new_x)
true_y = normalize_img(true_y)

generator.eval()
new_windows = split_for_patches(new_x, patch_size=(128, 128, 128), overlap=0.8)
new_y_hat_ls = predict_through_image_window(windows=new_windows, generator=generator, nz=args.nz, device=args.device)
new_y_hat = combine_pathces(new_x, new_y_hat_ls)
new_y = new_y_hat.to('cpu')

show_2d_slice(new_x, true_y, new_y, 20, input_img=args.input)

plt.clf()

show_2d_slice(new_x, true_y, new_y, 60, input_img=args.input)

plt.clf()

show_2d_slice(new_x, true_y, new_y, 100, input_img=args.input)
