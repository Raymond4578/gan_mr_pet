import time
import argparse
import numpy as np
import random
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from model import UNet, PatchGAN
from model import UnetGenerator
from utils import get_data_dir, load_data, split_data, get_data_batch
from utils import normalize_img
from utils import split_for_patches, combine_pathces
from utils import set_requires_grad
from utils import predict_through_image_window
from utils import get_ssim, get_psnr, get_rmse, get_mae
from load_model import show_2d_slice

# 这个文件是训练Pix2pix原代码模型的文件

# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=5441, help='Random seed.')
parser.add_argument('--alpha', type=int, default=100, help='Weight for L1 loss for generator.')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=20, help='Number of batch size.')
parser.add_argument('--g_lr', type=float, default=0.0002, help='Initial generator learning rate.')
parser.add_argument('--d_lr', type=float, default=0.000001, help='Initial discriminator learning rate.')
parser.add_argument('--use_dropout', type=bool, default=True, help='Use dropout in U-Net.')
parser.add_argument('--input', type=str, default='mr', help='Setting the input to MR image or PET image.')
parser.add_argument('--device', type=int, default=1, help='Setting the device of GPU, choose 0 or 1.')

args = parser.parse_args() # 调用parse_args()方法进行解析；解析成功之后即可使用。# 这个是使用argparse模块时的必备行，将参数进行关联
# args现在是一个叫Namespace的object

device = args.device
args.cuda = not args.no_cuda and torch.cuda.is_available() # add a variable for aist(rgs, cuda=True

# set random seeds for the training for the validation
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# get the affine matrix to get nii file
data_dir = get_data_dir()
filename = data_dir + '/AV1451_PET_within_180_days/wADNI_002_S_0413_2017_06_21_PET_AV1451_PVC01_SUVR.nii'
affine_matrix = nib.load(filename).affine

# load the data, and get the training set, evaluation set and test set ready
dataloader = load_data()
print(f'number of pairs in the dataset: {len(dataloader)}')
# put the image data into batches
data_train, data_test = split_data(dataloader, 0.8)
data_train = get_data_batch(data_train, args.batch_size)
data_test = get_data_batch(data_test, args.batch_size * 1)

## Initialize generator and discriminator
generator = UnetGenerator(in_channels=1, out_channels=1, num_downs=6, ngf=64, use_dropout=args.use_dropout)
discriminator = PatchGAN()

# Set up Loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()

# If use coda, move all the data into GPU memory
if args.cuda:
    print('cuda used.')
    print(f'Using device {args.device}.')
    generator.cuda(device)
    discriminator.cuda(device)
    adversarial_loss.cuda(device)
    l1_loss.cuda(device)

# set up optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
# set up learning rate decay
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[300], gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[300], gamma=0.1)

def train(epoch, data_train, data_test):
    '''
    The whole traning process for each epoch
    '''
    for i, entry in enumerate(data_train):
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

        # 然后切割图像
        x_windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)
        y_windows = split_for_patches(y, patch_size=(128, 128, 128), overlap=0.8)

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

            ######################
            # 先计算所有需要的tensor
            ######################

            y_hat_window = generator(x_window)

            fake_data = torch.cat([x_window, y_hat_window], dim=1)
            real_data = torch.cat([x_window, y_window], dim=1)

            true_mat = torch.ones_like(discriminator(real_data))
            false_mat = torch.zeros_like(discriminator(fake_data))
            if args.cuda:
                true_mat = true_mat.to(f'cuda:{args.device}')
                false_mat = false_mat.to(f'cuda:{args.device}')

            ################
            # 然后计算G的Loss
            ################
            set_requires_grad([discriminator], False)

            g_loss_adv = adversarial_loss(discriminator(fake_data), true_mat)
            g_loss_l1 = l1_loss(y_hat_window, y_window)
            g_loss = g_loss_adv + args.alpha * g_loss_l1

            g_loss_ls.append(g_loss.item())

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            ################
            # 然后计算D的Loss
            ################
            set_requires_grad([discriminator], True)

            real_loss = adversarial_loss(discriminator(real_data), true_mat)
            fake_loss = adversarial_loss(discriminator(fake_data.detach()), false_mat)
            d_loss = real_loss + fake_loss

            d_loss_ls.append(d_loss.item())

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        generator.eval()
        # 现在出了对于patch的循环，我们重新把之前切成的块用目前对于所有patch训练过的generator重新predict y来计算这个batch的SSIM等
        y_hat_ls = predict_through_image_window(windows=x_windows, generator=generator, device=args.device)
        y_hat = combine_pathces(x, y_hat_ls)
        if args.cuda:
            y = y.cuda(args.device)
            y_hat = y_hat.to(f'cuda:{args.device}')
        generator.train()

        # --------------------------------------------------------------------------------
        # Calculate measures to evaluate the model on training set (RMSE, MAE, PSNR, SSIM)
        # --------------------------------------------------------------------------------

        ssim_result = get_ssim(y, y_hat)
        psnr_result = get_psnr(y, y_hat)
        mae_result = get_mae(y, y_hat)
        rmse_result = get_rmse(y, y_hat)

        # -------------------------
        # Print the measure results
        # -------------------------

        print(
            f'Epoch: {epoch + 1:03d}/{args.epochs},',
            f'Batch: {i + 1:03d}/{len(data_train)},',
            f'D Loss: {np.average(d_loss_ls):.4f},',
            f'G Loss: {np.average(g_loss_ls):.4f},',
            f'RMSE: {np.average(rmse_result):.4f},',
            f'MAE: {np.average(mae_result):.4f},',
            f'PSNR: {np.average(psnr_result):.4f},',
            f'SSIM: {np.average(ssim_result):.4f},',
            f'Time: {time.time() - t:.4f}s.'
        )

    # At the end of each epoch, do evaluation of the model on the evaluation set
    test(data_test, generator=generator, input_img=args.input)

def test(data_test, generator, input_img):
    generator.eval()

    rmse_ls = []
    mae_ls = []
    psnr_ls = []
    ssim_ls = []

    for i, entry in enumerate(data_test):
        PET_img = entry[0]
        MR_img = entry[1]

        if input_img == 'mr':
            x = MR_img.clone().detach()
            y = PET_img.clone().detach()
        elif input_img == 'pet':
            x = PET_img.clone().detach()
            y = MR_img.clone().detach()
        if args.cuda:
            x = x.to(f'cuda:{args.device}')

        x = normalize_img(x)
        y = normalize_img(y)

        x_windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)

        y_hat_ls = predict_through_image_window(windows=x_windows, generator=generator, device=args.device)
        y_hat = combine_pathces(x, y_hat_ls)

        if args.cuda:
            y = y.cuda(args.device)
            y_hat = y_hat.to(f'cuda:{args.device}')

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

    return rmse_ls, mae_ls, psnr_ls, ssim_ls

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train(epoch, data_train, data_test)
    print(f'This epoch takes {(time.time() - epoch_start_time) / 60:.2f}min. ({time.time() - epoch_start_time:.2f}s)')
    scheduler_G.step()
    scheduler_D.step()

torch.save(generator, f'./model/UNet3d_input_{args.input}_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt')

# 以下的部分就是输出一个作为例子看看结果的图片，可以没有，代码基本与load_model.py中一致的
if args.input == 'mr':
    new_x = data_test[2][1][0, :, :, :, :].unsqueeze(0).cuda(device)
    true_y = data_test[2][0][0, :, :, :, :].unsqueeze(0)
elif args.input == 'pet':
    new_x = data_test[2][0][0, :, :, :, :].unsqueeze(0).cuda(device)
    true_y = data_test[2][1][0, :, :, :, :].unsqueeze(0)
new_x = normalize_img(new_x)
true_y = normalize_img(true_y)

generator.eval()
new_windows = split_for_patches(new_x, patch_size=(128, 128, 128), overlap=0.8)
new_y_hat_ls = predict_through_image_window(windows=new_windows, generator=generator, device=args.device)
new_y_hat = combine_pathces(new_x, new_y_hat_ls)
new_y = new_y_hat.to('cpu')

show_2d_slice(new_x, true_y, new_y, 20, input_img=args.input)

plt.clf()

show_2d_slice(new_x, true_y, new_y, 60, input_img=args.input)

plt.clf()

show_2d_slice(new_x, true_y, new_y, 100, input_img=args.input)

# new_y_np = new_y.detach().squeeze().cpu().numpy()
# test_img = nib.Nifti1Image(new_y_np, affine=affine_matrix)
# nib.save(test_img, f'test_output_input_{args.input}_{time.strftime("%y%m%d%H%M")}.nii')