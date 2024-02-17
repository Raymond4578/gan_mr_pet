import torch
import random
import numpy as np


from utils import load_data, split_data, get_data_batch
from utils import normalize_img
from utils import get_mae, get_rmse, get_ssim, get_psnr
from utils import split_for_patches, combine_pathces
from utils import predict_through_image_window

# 这个文件干的事情就是重复以下给model test performance的代码，跟train.py里一样的

def test(data_test, generator, input_img, device):
    generator.eval()

    rmse_ls = []
    mae_ls = []
    psnr_ls = []
    ssim_ls = []

    for i, entry in enumerate(data_test):
        PET_img = entry[0]
        MR_img = entry[1]

        # get ready input x and output y
        if input_img == 'mr':  # do MR to PET training
            x = MR_img.clone().detach()
            y = PET_img.clone().detach()
        elif input_img == 'pet':  # do PET to MR training
            x = PET_img.clone().detach()
            y = MR_img.clone().detach()

        if device == 0 or device == 1:
            x = x.cuda(device)

        x = normalize_img(x)
        y = normalize_img(y)

        # # 然后把它切成36份
        windows = split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8)

        y_hat_ls = predict_through_image_window(windows=windows, generator=generator, nz=8, device=device)
        y_hat = combine_pathces(x, y_hat_ls)

        if device == 0 or device == 1:
            y = y.cuda(device)
            y_hat = y_hat.cuda(device)

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

if __name__ == '__main__':
    seed = 5441
    device = -1
    batch_size = 1

    random.seed(seed)

    dataloader = load_data()
    print(f'number of pairs in the dataset: {len(dataloader)}')
    data_train, data_test = split_data(dataloader, 0.8)
    data_train = get_data_batch(data_train, batch_size)
    data_test = get_data_batch(data_test, batch_size * 1)

    print(f'The random seed is {seed}.')

    # generator_mr_to_pet = torch.load('./model/BPGAN_input_mr_seed_5441_2401231719.pt', map_location=f'cuda:{device}')
    # generator_mr_to_pet = torch.load('./model/BPGAN_input_mr_seed_4027_2401232236.pt', map_location=f'cuda:{device}')
    # generator_mr_to_pet = torch.load('./model/BPGAN_input_mr_seed_4446_2401250251.pt', map_location=f'cuda:{device}')
    # generator_mr_to_pet = torch.load('./model/BPGAN_input_mr_seed_4025_2401251607.pt', map_location=f'cuda:{device}')
    # generator_mr_to_pet = torch.load('./model/BPGAN_input_mr_seed_3023_2401260126.pt', map_location=f'cuda:{device}')
    # generator_mr_to_pet = torch.load('./model/BPGAN_orig_input_mr_seed_5441_2402061307.pt', map_location=f'cuda:{device}')
    generator_mr_to_pet = torch.load('./model/BPGAN_orig_input_mr_seed_5441_2402061307.pt',
                                     map_location=f'cpu')
    print('For MR to PET task:')
    test(data_test, generator_mr_to_pet, input_img='mr', device=device)

    # generator_pet_to_mr = torch.load('./model/BPGAN_input_pet_seed_5441_2401231121.pt', map_location=f'cuda:{device}')
    # generator_pet_to_mr = torch.load('./model/BPGAN_input_pet_seed_4027_2401242122.pt', map_location=f'cuda:{device}')
    # generator_pet_to_mr = torch.load('./model/BPGAN_input_pet_seed_4446_2401250540.pt', map_location=f'cuda:{device}')
    # generator_pet_to_mr = torch.load('./model/BPGAN_input_pet_seed_4025_2401251858.pt', map_location=f'cuda:{device}')
    # generator_pet_to_mr = torch.load('./model/BPGAN_input_pet_seed_3023_2401260417.pt', map_location=f'cuda:{device}')
    # generator_pet_to_mr = torch.load('./model/BPGAN_orig_input_pet_seed_5441_2402040835.pt', map_location=f'cuda:{device}')
    generator_pet_to_mr = torch.load('./model/BPGAN_orig_input_pet_seed_5441_2402040835.pt',
                                     map_location=f'cpu')
    print('For PET to MR task:')
    test(data_test, generator_pet_to_mr, input_img='pet', device=device)