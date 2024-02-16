import time
import argparse
import numpy as np
import random
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from model import UNet, PatchGAN
from utils import load_data, split_data, get_data_batch
from utils import get_data_dir
from utils import normalize_img
from utils import get_ssim, get_psnr, get_rmse, get_mae
from load_model import show_2d_slice

# 这个文件是训练Adversarial U-Net自己写的代码模型的文件

# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=5441, help='Random seed.')
parser.add_argument('--alpha', type=int, default=100, help='Weight for L1 loss for generator.')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train.')
parser.add_argument('--niter', type=int, default=80, help='Number of epochs before learning rate decay.')
parser.add_argument('--batch_size', type=int, default=2, help='Number of batch size.')
parser.add_argument('--g_lr', type=float, default=0.0002, help='Initial generator learning rate.')
parser.add_argument('--d_lr', type=float, default=0.00001, help='Initial discriminator learning rate.')
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
generator = UNet()
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
def get_scheduler(optimizer, args):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.epochs - args.niter + 1)
        return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

scheduler_G = get_scheduler(optimizer_G, args)
scheduler_D = get_scheduler(optimizer_D, args)

def train(epoch, data_train, data_test):
    '''
    The whole traning process for each epoch
    '''
    for i, entry in enumerate(data_train):
        # 每次训练一个batch的数据
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

        # Move data to GPU memory
        if args.cuda:
            x = x.cuda(device)
            y = y.cuda(device)

        # ---------------
        # Train Generator
        # ---------------

        # set generator optimizer to zero gradient
        optimizer_G.zero_grad()

        # get the predicted y
        y_hat = generator(x)

        # combine x and y_hat and x and y as pair
        paired_hat = torch.cat([x, y_hat], dim=1)
        paired_true = torch.cat([x, y], dim=1)

        # get ground choose
        y_true = torch.ones_like(discriminator(paired_true))
        y_false = torch.zeros_like(discriminator(paired_hat))
        # move ground choose data to GPU memory
        if args.cuda:
            y_true = y_true.cuda(device)
            y_false = y_false.cuda(device)

        # get the loss for training
        g_loss_adv = adversarial_loss(discriminator(paired_hat.detach()), y_true) # adversarial loss
        g_loss_l1 = l1_loss(y_hat, y) # l1 loss
        g_loss = g_loss_adv + args.alpha * g_loss_l1 # calculate the final loss for generator

        # backward propagation
        g_loss.backward()
        torch.nn.utils.clip_grad_value_(generator.parameters(), 1) # Do gradient clipping for a stable training
        optimizer_G.step() # do backward propagation

        # -------------------
        # Train Discriminator
        # -------------------

        # set discriminator optimizer to zero gradient
        optimizer_D.zero_grad()

        # measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(paired_true), y_true)
        # 使用detach方法将张量从计算图中分离，使得在计算损失时不会影响该张量的梯度。换句话说，gen_imgs.detach() 返回一个新的张量，它不再与生成器的计算图相关联。
        fake_loss = adversarial_loss(discriminator(paired_hat.detach()), y_false)
        d_loss = (real_loss + fake_loss) / 2

        # backward propagation
        d_loss.backward()
        optimizer_D.step() # do backward propagation

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
            f'D Loss: {d_loss.item():.4f},',
            f'G Loss: {g_loss.item():.4f},',
            f'RMSE: {np.average(rmse_result):.4f},',
            f'MAE: {np.average(mae_result):.4f},',
            f'PSNR: {np.average(psnr_result):.4f},',
            f'SSIM: {np.average(ssim_result):.4f},',
            f'Time: {time.time() - t:.4f}s.'
        )

    # At the end of each epoch, do evaluation of the model on the evaluation set
    test(data_test)

def test(data_test):
    generator.eval()

    rmse_ls = []
    mae_ls = []
    psnr_ls = []
    ssim_ls = []

    for i, entry in enumerate(data_test):
        PET_img = entry[0]
        MR_img = entry[1]

        if args.input == 'mr':
            x = MR_img.clone().detach()
            y = PET_img.clone().detach()
        elif args.input == 'pet':
            x = PET_img.clone().detach()
            y = MR_img.clone().detach()
        if args.cuda:
            x = x.cuda(device)
            y = y.to(f'cuda:{device}')

        x = normalize_img(x)
        y = normalize_img(y)

        with torch.no_grad():
            y_hat = generator(x)

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

test(data_test)

torch.save(generator, f'./model/UNet3d_my_input_{args.input}_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt')

# 以下的部分就是输出一个作为例子看看结果的图片，可以没有，代码基本与load_model.py中一致的
if args.input == 'mr':
    new_x = data_test[2][1][0, :, :, :, :].unsqueeze(0).cuda(device)
    true_y = data_test[2][0][0, :, :, :, :].unsqueeze(0)
elif args.input == 'pet':
    new_x = data_test[2][0][0, :, :, :, :].unsqueeze(0).cuda(device)
    true_y = data_test[2][1][0, :, :, :, :].unsqueeze(0)

new_x = normalize_img(new_x)
true_y = normalize_img(true_y)

with torch.no_grad():
    new_y = generator(new_x).to('cpu')
# print(new_y.shape)

show_2d_slice(new_x, true_y, new_y, 20, input_img=args.input)

plt.clf()

show_2d_slice(new_x, true_y, new_y, 60, input_img=args.input)

plt.clf()

show_2d_slice(new_x, true_y, new_y, 100, input_img=args.input)

# new_y_np = new_y.detach().squeeze().cpu().numpy()
# test_img = nib.Nifti1Image(new_y_np, affine=affine_matrix)
# nib.save(test_img, f'test_output_input_{args.input}_{time.strftime("%y%m%d%H%M")}.nii')