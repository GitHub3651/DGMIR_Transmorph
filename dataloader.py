import nibabel as nib
import torch
import torch.utils.data as data
import numpy as np


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


class GetLoader_Brats2018(data.Dataset):
    def __init__(self, data_list, vol_size):
        fd = open(data_list, 'r')
        data_list = fd.read().splitlines()
        fd.close()
        self.n_data = len(data_list)
        self.data_list = data_list
        self.vol_size = vol_size

    def __getitem__(self, item):
        # BraTS2018 Dataset
        fixed_seg = nib.load(self.data_list[item * 4])
        moving_seg = nib.load(self.data_list[item * 4 + 1])
        fixed = nib.load(self.data_list[item * 4 + 2])
        moving = nib.load(self.data_list[item * 4 + 3])

        fixed = np.array(fixed.dataobj)
        moving = np.array(moving.dataobj)
        fixed_seg = np.array(fixed_seg.dataobj)
        moving_seg = np.array(moving_seg.dataobj)

        # crop image
        fixed = center_crop(fixed, self.vol_size)
        moving = center_crop(moving, self.vol_size)
        fixed_seg = center_crop(fixed_seg, self.vol_size)
        moving_seg = center_crop(moving_seg, self.vol_size)

        # Norm
        if np.mean(fixed) > 1:
            fixed = imgnorm(fixed)
            moving = imgnorm(moving)

        fixed = torch.tensor(fixed, dtype=torch.float32, device='cuda')
        moving = torch.tensor(moving, dtype=torch.float32, device='cuda')
        fixed_seg = torch.from_numpy(fixed_seg).float().cuda()
        moving_seg = torch.from_numpy(moving_seg).float().cuda()

        return fixed, moving, fixed_seg, moving_seg

    def __len__(self):
        return self.n_data


def center_crop(data, target_size):
    """
    Perform center cropping of a 3D numpy array.

    Args:
        data (numpy.ndarray): The input 3D array to crop.
        target_size (tuple): The target size (depth, height, width).

    Returns:
        numpy.ndarray: The center-cropped array.
    """
    depth, height, width = data.shape
    target_depth, target_height, target_width = target_size

    start_d = (depth - target_depth) // 2
    start_h = (height - target_height) // 2
    start_w = (width - target_width) // 2

    return data[
           start_d:start_d + target_depth,
           start_h:start_h + target_height,
           start_w:start_w + target_width
           ]


def load_4D_with_crop2(X, cropx, cropy, cropz):
    # X = nib.load(name)
    # X = X.get_fdata()

    x, y, z = X.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    startz = z // 2 - cropz // 2

    X = X[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz]

    X = np.reshape(X, X.shape)
    return X
