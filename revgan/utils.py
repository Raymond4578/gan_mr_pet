import os
import random
import torch
import numpy as np
from monai.inferers import SlidingWindowSplitter
from monai.metrics import MAEMetric, RMSEMetric, PSNRMetric
from monai.metrics.regression import SSIMMetric

from dataloader import ADNI_AV1451_Dataset

def get_data_dir():
    '''
    Change the code for your data path in the server
    记得改代码（写数据的读取路径）
    '''
    # home_dir = os.path.expanduser('~')
    # data_dir = os.path.join(home_dir, 'gan', 'data')
    cwd = os.getcwd()
    home_dir = os.path.dirname(cwd)
    data_dir = os.path.join(home_dir, 'data')
    return data_dir

def load_data():
    data_dir = get_data_dir()
    PET_dir = os.path.join(data_dir, 'AV1451_PET_within_180_days/')
    MR_dir = os.path.join(data_dir, 'AV1451_MR_within_180_days/')

    PET_name = [filename for filename in os.listdir(PET_dir) if filename.endswith(".nii")]
    MR_name = [filename for filename in os.listdir(MR_dir) if filename.endswith(".nii")]

    dataloader = ADNI_AV1451_Dataset(PET_dir, MR_dir, PET_name, MR_name)

    return dataloader


def split_data(dataloader, p):
    # 随机分训练测试集
    dataloader = list(dataloader)
    random.shuffle(dataloader)
    split_point = int(len(dataloader) * p) # get test sample size
    return dataloader[:split_point], dataloader[split_point:]

def get_data_batch(dataloader, batch_size=4):
    '''
    手动把数据放入一个个batch
    '''
    data = []
    for i, entry in enumerate(dataloader):
        if i == 0:
            PET_img = entry['PET'].unsqueeze(0)
            MR_img = entry['MR'].unsqueeze(0)
        elif i % batch_size == 0:
            data.append((PET_img, MR_img))
            PET_img = entry['PET'].unsqueeze(0)
            MR_img = entry['MR'].unsqueeze(0)
        elif i == len(dataloader) - 1:
            PET_img = torch.cat([PET_img, entry['PET'].unsqueeze(0)], dim=0)
            MR_img = torch.cat([MR_img, entry['MR'].unsqueeze(0)], dim=0)
            data.append((PET_img, MR_img))
        else:
            PET_img = torch.cat([PET_img, entry['PET'].unsqueeze(0)], dim=0)
            MR_img = torch.cat([MR_img, entry['MR'].unsqueeze(0)], dim=0)

    return data

def normalize_img(x):
    '''
    normalized the data into [-1, 1]
    '''
    x_reshaped = x.reshape(x.size(0), x.size(1), -1)
    max_value = x_reshaped.max(dim=2, keepdim=True)[0]
    min_value = x_reshaped.min(dim=2, keepdim=True)[0]
    x_normalized = 2 * (x_reshaped - min_value) / (max_value - min_value) - 1
    # print(x_normalized.max(dim=2, keepdim=True)[0])
    x_normalized = x_normalized.view_as(x)
    return x_normalized

def split_for_patches(x, patch_size=(128, 128, 128), overlap=0.8):
    '''
    把一个batch中的datq切成好几份
    '''
    # 然后把它切成几份
    splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=overlap)
    windows = list(splitter(x))

    return windows

def combine_pathces(x, windows):
    # 初始化输出图像和重叠计数
    output_image = torch.zeros_like(x).to('cpu')
    overlap_count = torch.zeros_like(x).to('cpu')

    for y_hat_1, idx in windows:
        y_hat_1 = y_hat_1.to('cpu')
        z_start, y_start, x_start = idx
        z_stop = min(z_start + y_hat_1.shape[2], x.shape[2])
        y_stop = min(y_start + y_hat_1.shape[3], x.shape[3])
        x_stop = min(x_start + y_hat_1.shape[4], x.shape[4])
        # 累加块并更新重叠计数
        output_image[..., z_start:z_stop, y_start:y_stop, x_start:x_stop] += y_hat_1[..., :z_stop - z_start,
                                                                             :y_stop - y_start, :x_stop - x_start]
        overlap_count[:, :, z_start:z_stop, y_start:y_stop, x_start:x_stop] += 1

    # 计算平均值
    output_image /= overlap_count

    return output_image

def set_requires_grad(nets, requires_grad=False):
    '''
    Set whether to update the gradient of a specific model
    '''
    if not isinstance(nets, list):
        nets = list(nets)
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def predict_through_image_window(windows, generator, direction, device):
    '''
    Do prediction patchwize
    '''
    y_hat_ls = []

    for j, window in enumerate(windows):
        x_window, idx = window
        x_window = x_window.cuda(device)

        with torch.no_grad():
            if direction == 'fwd':
                y_hat_for_combine = generator(x_window)
            elif direction == 'bwd':
                y_hat_for_combine = generator(x_window, True)
        y_hat_ls.append((y_hat_for_combine, idx))

    return y_hat_ls

def get_ssim(y, y_hat):
    '''
    :param y: True image
    :param y_hat: Fake image
    :return: SSIM value
    '''
    ssim_metric = SSIMMetric(spatial_dims=3)
    ssim_values = ssim_metric(y_pred=y_hat, y=y)
    return list(ssim_values.view(-1).to('cpu').numpy())

def get_psnr(y, y_hat):
    '''
    :param y: True image
    :param y_hat: Fake image
    :return: PSNR value
    '''
    psnr_metric = PSNRMetric(max_val=2)
    psnr_values = psnr_metric(y_pred=y_hat, y=y)
    return list(psnr_values.view(-1).to('cpu').numpy())

def get_mae(y, y_hat):
    '''
    :param y: True image
    :param y_hat: Fake image
    :return: MAE value
    '''

    mae_metric = MAEMetric()
    mae_values = mae_metric(y_pred=y_hat, y=y)
    return list(mae_values.view(-1).to('cpu').numpy())

def get_rmse(y, y_hat):
    '''
    :param y: True image
    :param y_hat: Fake image
    :return: RMSE value
    '''

    rmse_metric = RMSEMetric()
    rmse_values = rmse_metric(y_pred=y_hat, y=y)
    return list(rmse_values.view(-1).to('cpu').numpy())

if __name__ == '__main__':
    device = 0
    y = torch.randn(4, 1, 121, 145, 121).to(f'cuda:{device}')
    y_norm = normalize_img(y)
    print(y_norm.min())