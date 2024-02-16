import torch
import numpy as np
import random

from utils import load_data, split_data, get_data_batch
from utils import normalize_img
from utils import split_for_patches, combine_pathces
from utils import predict_through_image_window
from utils import get_rmse, get_mae, get_psnr, get_ssim

# 这个文件干的事情就是重复以下给model test performance的代码，跟train.py里一样的
# 注意！我没有写我自己写的模型的result.py文件！这个script只适用Pix2pix原代码版本的模型

def test(data_test, generator, input_img, device):
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

        x = x.to(f'cuda:{device}')
        y = y.to(f'cuda:{device}')

        x = normalize_img(x)
        y = normalize_img(y)

        x_windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)

        y_hat_ls = predict_through_image_window(windows=x_windows, generator=generator, device=device)
        y_hat = combine_pathces(x, y_hat_ls)

        y = y.cuda(device)
        y_hat = y_hat.to(f'cuda:{device}')

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

if __name__ == '__main__':
    seed = 5441
    batch_size = 1
    device = 1

    random.seed(seed)

    dataloader = load_data()
    print(f'number of pairs in the dataset: {len(dataloader)}')
    data_train, data_test = split_data(dataloader, 0.8)
    data_train = get_data_batch(data_train, batch_size)
    data_test = get_data_batch(data_test, batch_size * 1)

    generator_mr_to_pet = torch.load('./model/UNet3d_new_input_mr_seed_5441_2401302259.pt', map_location=f'cuda:{device}')
    print('For MR to PET task:')
    test(data_test, generator_mr_to_pet, input_img='mr', device=device)

    generator_pet_to_mr = torch.load('./model/UNet3d_new_input_pet_seed_5441_2401301916.pt', map_location=f'cuda:{device}')
    print('For PET to MR task:')
    test(data_test, generator_pet_to_mr, input_img='pet', device=device)
