import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from utils import load_data, split_data, get_data_batch
from utils import split_for_patches, combine_pathces
from utils import normalize_img
from utils import predict_through_image_window

# 这个文件就是给模型的虚拟啦输出的参考图，跟train.py内的内容基本上是一样的

def show_2d_slice(x, y, y_hat, x_hat, slice_idx, loss=0):
    y_hat = y_hat.detach()[0, 0, :, :, slice_idx].numpy()
    x_hat = x_hat.detach()[0, 0, :, :, slice_idx].numpy()
    y = y.cpu()[0, 0, :, :, slice_idx].numpy()
    x = x.cpu()[0, 0, :, :, slice_idx].numpy()

    max_y = max(np.max(y), np.max(y_hat))
    min_y = min(np.min(y), np.min(y_hat))
    max_x = max(np.max(x), np.max(x_hat))
    min_x = min(np.min(x), np.min(x_hat))

    plt.subplot(2, 2, 1)
    plt.imshow(y, cmap='gray')
    plt.colorbar()
    plt.clim(min_y, max_y)
    plt.title('Real PET')
    plt.subplot(2, 2, 2)
    plt.imshow(y_hat, cmap='gray')
    plt.colorbar()
    plt.clim(min_y, max_y)
    plt.title(f'Estimated PET (SSIM: {ssim(y, y_hat):.4f})')
    plt.subplot(2, 2, 3)
    plt.imshow(x, cmap='gray')
    plt.colorbar()
    plt.clim(min_x, max_x)
    plt.title('Real MR')
    plt.subplot(2, 2, 4)
    plt.imshow(x_hat, cmap='gray')
    plt.colorbar()
    plt.clim(min_x, max_x)
    plt.title(f'Estimated MR (SSIM: {ssim(x, x_hat):.4f})')
    plt.tight_layout()
    plt.savefig(f'testimg_loss_{loss}_{slice_idx}.png')

if __name__ == '__main__':
    generator = torch.load('./model/RevGAN_loss_0_seed_5441_2401300920.pt')

    device = 0
    loss = 0

    dataloader = load_data()
    print(f'number of pairs in the dataset: {len(dataloader)}')
    data_train, data_test = split_data(dataloader, 0.8)
    data_train = get_data_batch(data_train, 47)
    data_test = get_data_batch(data_test, 47)

    true_x = data_test[0][1][0, :, :, :, :].unsqueeze(0).cuda(device)
    true_y = data_test[0][0][0, :, :, :, :].unsqueeze(0).cuda(device)
    true_x = normalize_img(true_x)
    true_y = normalize_img(true_y)

    generator.eval()
    new_x_windows = split_for_patches(true_x, patch_size=(32, 32, 32), overlap=0.05)
    new_y_windows = split_for_patches(true_y, patch_size=(32, 32, 32), overlap=0.05)
    new_y_hat_ls = predict_through_image_window(windows=new_x_windows, generator=generator, direction='fwd',
                                                device=device)
    new_x_hat_ls = predict_through_image_window(windows=new_y_windows, generator=generator, direction='bwd',
                                                device=device)
    new_y_hat = combine_pathces(true_x, new_y_hat_ls)
    new_x_hat = combine_pathces(true_y, new_x_hat_ls)
    new_y = new_y_hat.to('cpu')
    new_x = new_x_hat.to('cpu')

    show_2d_slice(true_x, true_y, new_y, new_x, 20, loss=loss)

    plt.clf()

    show_2d_slice(true_x, true_y, new_y, new_x, 60, loss=loss)

    plt.clf()

    show_2d_slice(true_x, true_y, new_y, new_x, 100, loss=loss)