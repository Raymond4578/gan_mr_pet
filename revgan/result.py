import torch
import random
import numpy as np

from utils import load_data, split_data, get_data_batch, get_mae, get_rmse, get_ssim, get_psnr
from utils import normalize_img
from utils import split_for_patches, combine_pathces, predict_through_image_window

# 这个文件干的事情就是重复以下给model test performance的代码，跟train.py里一样的

def test(data_test, generator, PET_to_MR, device, orig=False):
    rmse_ls = []
    mae_ls = []
    psnr_ls = []
    ssim_ls = []

    for i, entry in enumerate(data_test):
        PET_img = entry[0]
        MR_img = entry[1]

        if not PET_to_MR:
            x = MR_img.clone().detach()
            y = PET_img.clone().detach()
        else:
            y = MR_img.clone().detach()
            x = PET_img.clone().detach()

        x = normalize_img(x)
        y = normalize_img(y)

        if device == 0 or device == 1:
            x = x.to(f'cuda:{device}')
            y = y.to(f'cuda:{device}')

        if orig:
            new_windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)
            if not PET_to_MR:
                new_y_hat_ls = predict_through_image_window(windows=new_windows, generator=generator, direction='fwd',
                                                        device=device)
            else:
                new_y_hat_ls = predict_through_image_window(windows=new_windows, generator=generator, direction='bwd',
                                                            device=device)
            y_hat = combine_pathces(x, new_y_hat_ls)
        else:
            if not PET_to_MR:
                y_hat = generator(x)
            else:
                y_hat = generator(x, True)

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
    # print(f'The average SSIM for test data is {np.average(ssim_ls):.4f}.')
    generator.train()
    return ssim_ls

random.seed(5441)

device = -1
loss = 0

dataloader = load_data()
print(f'number of pairs in the dataset: {len(dataloader)}')
data_train, data_test = split_data(dataloader, 0.8)
data_train = get_data_batch(data_train, 1)
data_test = get_data_batch(data_test, 1)

if device == 0 or device == 1:
    generator = torch.load('./model/RevGAN_my_loss_0_seed_5441_2402051529.pt', map_location=f'cuda:{device}')
    # generator = torch.load('./model/RevGAN_orig_loss_0_seed_5441_2402021115.pt', map_location=f'cuda:{device}')
else:
    generator = torch.load('./model/RevGAN_my_loss_0_seed_5441_2402051529.pt', map_location=f'cpu')
    # generator = torch.load('./model/RevGAN_orig_loss_0_seed_5441_2402021115.pt', map_location=f'cpu')

print('For MR to PET task:')
test(data_test, generator, PET_to_MR=False, device=device)
print('For PET to MR task:')
test(data_test, generator, PET_to_MR=True, device=device)