import nibabel as nib
import torch.utils.data
import time
import os
from metric import *
from dataloader import imgnorm, center_crop, GetLoader_Brats2018, load_4D_with_crop2
from model2 import DGMIR
from losses import dicegup
from functions import SpatialTransformer2
from metric import compute_surface_distances, compute_robust_hausdorff


def compute_smoothness(deformation_field):
    # Compute gradients
    gradients = torch.stack(torch.gradient(deformation_field), dim=-1)

    # Compute squared gradients
    squared_gradients = gradients ** 2

    # Sum over dimensions
    sum_squared_gradients = torch.sum(squared_gradients, dim=-1)

    # Compute mean smoothness
    mean_smoothness = torch.mean(sum_squared_gradients)

    return mean_smoothness


def test_all(model_path, file_path, ckpt, vol_size, cls_num):
    # define Network
    net = DGMIR(vol_size).cuda()
    stn = SpatialTransformer2().cuda()

    # load dataset
    dataset_target = GetLoader_Brats2018(file_path, vol_size)
    dataloader_target = torch.utils.data.DataLoader(dataset=dataset_target, batch_size=1)
    len_dataloader = len(dataloader_target) // 4
    data_target_iter = iter(dataloader_target)

    # load ckpt file
    checkpoint = torch.load(os.path.join(model_path, ckpt))
    net.load_state_dict(checkpoint['model_state_dict'])
    # print(net)
    net.eval()

    njd_mean = []
    njd_std = []
    hd95_result = []
    dice_result = []

    with torch.no_grad():
        for i in range(len_dataloader):

            data_target = data_target_iter.__next__()
            fixed, moving, fixed_seg, moving_seg = data_target

            fixed = torch.unsqueeze(fixed, dim=1)
            moving = torch.unsqueeze(moving, dim=1)
            fixed_seg = torch.unsqueeze(fixed_seg, dim=1)
            moving_seg = torch.unsqueeze(moving_seg, dim=1)

            net.zero_grad()
            flow, warped = net(fixed, moving, 'test')

            njd = jacobian_determinant_gpu(flow.permute(0, 1, 4, 3, 2)).cpu().numpy()
            njd_mean.append(np.mean(njd < 0))
            njd_std.append(np.std(njd))
            warped_seg = stn(moving_seg, flow, mode='nearest')[0, 0, :, :, :]
            fixed_seg = fixed_seg[0, 0, :, :, :]
            dice_score_reg = dicegup(warped_seg, fixed_seg)
            dice_score_reg = dice_score_reg.cpu().numpy()
            dice_result.append(dice_score_reg)

            warped_seg = warped_seg.cpu().numpy()
            fixed_seg = fixed_seg.cpu().numpy()
            count = 0
            hd95 = 0
            for i in range(1, cls_num):
                if ((fixed_seg == i).sum() == 0) or ((warped_seg == i).sum() == 0):
                    continue
                hd95 += compute_robust_hausdorff(compute_surface_distances((fixed_seg == i), (warped_seg == i), np.ones(3)), 95.)
                count += 1
            hd95 /= count
            hd95_result.append(hd95)

    print("NJD_mean: %.6f, NJD_std: %.6f, HD95: %.6f±%.6f, DICE: %.6f±%.6f" % (np.mean(njd_mean), np.mean(njd_std), np.mean(hd95_result), np.std(hd95_result), np.mean(dice_result), np.std(dice_result)))


if __name__ == '__main__':
    batch = 1
    vol_size = [160, 192, 128]
    cls_num = 29
    model_path = './checkpoints/BraTS2018/DGMIR'
    test_path = "./data/BraTS2018_test.txt"
    ckpt_name = 'DGMIR.pth'

    test_all(model_path, test_path, ckpt_name, vol_size, cls_num)
