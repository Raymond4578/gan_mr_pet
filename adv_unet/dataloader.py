import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class ADNI_AV1451_Dataset(Dataset):

    def __init__(self, PET_image_dir, MR_image_dir, PET_image_names, MR_image_names, transform=None):
        print("Initializing dataset")
        self.MR_image_dir = MR_image_dir
        self.PET_image_dir = PET_image_dir
        self.PET_image_names = PET_image_names  # [filename for filename in os.listdir(PET_image_dir) if filename.endswith(".nii")]
        self.MR_image_names = MR_image_names  # [filename for filename in os.listdir(MR_image_dir) if filename.endswith(".nii")]
        self.PET_image_names.sort()
        self.MR_image_names.sort()

        self.transform = transform

    def __len__(self):
        return len(self.PET_image_names)

    def __getitem__(self, idx): # get item
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # obtain the ith paired image names
        ith_PET_name = self.PET_image_names[idx]
        ith_PET_name_splits = ith_PET_name.split('_')
        ith_MR_name = self.MR_image_names[idx]
        ith_MR_name_splits = ith_MR_name.split('_')

        # check PET & MR match
        ith_PET_time = ith_PET_name_splits[4] + ith_PET_name_splits[5] + ith_PET_name_splits[6]
        ith_PET_MR_time = ith_MR_name_splits[5][3:] + ith_MR_name_splits[6] + ith_MR_name_splits[7]

        if ith_PET_time != ith_PET_MR_time:
            print("PET time and PET_MR time not consistent")

        PET_image = np.array(nib.load(self.PET_image_dir + ith_PET_name).dataobj)
        # aff_mat = nib.load(self.PET_image_dir + ith_PET_name).affine
        # print(aff_mat)
        MR_image = np.array(nib.load(self.MR_image_dir + ith_MR_name).dataobj)

        PET_image = PET_image[None]  # add (empty) channel dimension
        MR_image = MR_image[None]
        PET_image = torch.FloatTensor(PET_image)
        MR_image = torch.FloatTensor(MR_image)

        sample = {'PET': PET_image, 'MR': MR_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
