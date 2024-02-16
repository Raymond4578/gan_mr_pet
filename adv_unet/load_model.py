import time
import torch
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from utils import load_data, get_data_batch, split_data
from utils import normalize_img
from utils import split_for_patches, combine_pathces
from utils import predict_through_image_window

# 这个文件就是给模型的虚拟啦输出的参考图，跟train.py内的内容基本上是一样的

def show_2d_slice(x, y, y_hat, slice_idx, input_img='mr'):
    # extract the 2D image for x, y and y_hat
    y_hat = y_hat.detach()[0, 0, :, :, slice_idx].numpy()
    y = y[0, 0, :, :, slice_idx].numpy()
    x = x.cpu()[0, 0, :, :, slice_idx].numpy()

    # then plot all the images
    plt.subplot(2, 2, 1)
    plt.imshow(y_hat, cmap='gray')
    plt.colorbar()
    plt.clim(-1, 1)
    if input_img == 'mr':
        plt.title('Estimated PET')
    elif input_img == 'pet':
        plt.title('Estimated MR')
    plt.subplot(2, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.colorbar()
    plt.clim(-1, 1)
    if input_img == 'mr':
        plt.title(f'Real PET (SSIM: {ssim(y, y_hat):.4f})')
    elif input_img == 'pet':
        plt.title(f'Real MR (SSIM: {ssim(y, y_hat):.4f})')
    plt.subplot(2, 2, 3)
    plt.imshow(x, cmap='gray')
    plt.colorbar()
    plt.clim(-1, 1)
    if input_img == 'mr':
        plt.title(f'Real MR (SSIM: {ssim(x, y_hat):.4f})')
    elif input_img == 'pet':
        plt.title(f'Real PET (SSIM: {ssim(x, y_hat):.4f})')
    plt.tight_layout()
    plt.savefig(f'testimg_input_{input_img}_{slice_idx}_{time.strftime("%y%m%d%H%M")}.png')

if __name__ == '__main__':
    # This is for generating a sample of result
    seed = 5441
    batch_size = 20
    device = 0
    input_img = 'pet'

    random.seed(seed)

    if input_img == 'mr':
        generator = torch.load('./model/UNet3d_new_input_mr_seed_5441_2401302259.pt').to(f'cuda:{device}')
    elif input_img == 'pet':
        generator = torch.load('./model/UNet3d_new_input_pet_seed_5441_2401301916.pt').to(f'cuda:{device}')
    # generator = torch.load('./PET_MR_result/UNet3d_2312181421.pt').cuda(device)

    dataloader = load_data()
    print(f'number of pairs in the dataset: {len(dataloader)}')
    data_train, data_test = split_data(dataloader, 0.8)
    data_train = get_data_batch(data_train, batch_size)
    data_test = get_data_batch(data_test, batch_size * 1)

    if input_img == 'mr':
        new_x = data_test[0][1][0, :, :, :, :].unsqueeze(0).cuda(device)
        true_y = data_test[0][0][0, :, :, :, :].unsqueeze(0)
    elif input_img == 'pet':
        new_x = data_test[0][0][0, :, :, :, :].unsqueeze(0).cuda(device)
        true_y = data_test[0][1][0, :, :, :, :].unsqueeze(0)

    new_x = normalize_img(new_x)
    true_y = normalize_img(true_y)

    generator.eval()
    new_windows = split_for_patches(new_x, patch_size=(128, 128, 128), overlap=0.8)
    new_y_hat_ls = predict_through_image_window(windows=new_windows, generator=generator, device=device)
    new_y_hat = combine_pathces(new_x, new_y_hat_ls)
    new_y = new_y_hat.to('cpu')

    # This commented part is for example of the model
    show_2d_slice(new_x, true_y, new_y, 20)

    plt.clf()

    show_2d_slice(new_x, true_y, new_y, 60)

    plt.clf()

    show_2d_slice(new_x, true_y, new_y, 100)

    # y_hat = new_y.detach()[0, 0, :, :, 61].numpy()
    # y = true_y[0, 0, :, :, 61].numpy()
    # max_value = max(np.max(y), np.max(y_hat))
    #
    # plt.imshow(y_hat, cmap='gray')
    # plt.colorbar()
    # plt.clim(0, max_value)
    # plt.title(f'3D Adversarial U-Net (SSIM: {ssim(y, y_hat):.4f})')
    # plt.savefig(f'fake_pet_3d.png')
    # # plt.savefig(f'fake_mr_3d.png')
    #
    # plt.clf()
    #
    # plt.imshow(y, cmap='gray')
    # plt.colorbar()
    # plt.clim(0, max_value)
    # plt.title(f'True PET')
    # plt.savefig(f'true_pet.png')
    # # plt.title(f'True MR')
    # # plt.savefig(f'true_mr.png')
    #
    # with open('saved_test_sample_mr_pet.pkl', 'wb') as file:
    # # with open('saved_test_sample_pet_mr.pkl', 'wb') as file:
    #     pkl.dump([new_x.cpu(), true_y], file)



