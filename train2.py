import torch.optim as optim
import torch.utils.data
import numpy as np
import time
import nibabel as nib
from dataloader import GetLoader_Brats2018
from model2 import DGMIR
from losses import MINDSSCLoss, Grad, ncc_loss, dicegup, mask_to_one_hot, compute_per_channel_dice
from functions import SpatialTransformer2


def save_tensor_as_nii(tensor, filename):
    np_tensor = tensor.cpu().detach().numpy()
    nii_img = nib.Nifti1Image(np_tensor, np.eye(4))  # 使用单位矩阵作为仿射矩阵
    nib.save(nii_img, filename)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


if __name__ == '__main__':
    train_path = "./data/BraTS2018_train.txt"
    valid_path = "./data/BraTS2018_valid.txt"

    batch = 1
    n_epoch = 300
    lr = 1e-4
    vol_size = [160, 192, 128]
    weight = [1, 0.5, 0.5]  # Lsim, Lsmooth, Ldice
    cls_num = 29
    model_path = './checkpoints/'

    # load Dataset
    dataset_train = GetLoader_Brats2018(train_path, vol_size)
    dataset_valid = GetLoader_Brats2018(valid_path, vol_size)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch)
    dataloader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=batch)

    net = DGMIR(vol_size).cuda()
    stn = SpatialTransformer2().cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    image_loss_func = MINDSSCLoss()
    deformation_loss_func = Grad('l2')

    # load ckpt file for continuing training
    # checkpoint = torch.load(os.path.join(model_path, 'ckpt.pth'))
    # net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_last = checkpoint['epoch']

    best_record = 0.0
    epoch_last = 0

    for epoch in range(epoch_last, n_epoch):
        len_dataloader_train = len(dataloader_train) // 4
        len_dataloader_valid = len(dataloader_valid) // 4
        data_train_iter = iter(dataloader_train)
        data_valid_iter = iter(dataloader_valid)
        epoch_start_time = time.time()

        epoch_sim_loss = []
        epoch_grad_loss = []
        epoch_total_loss = []

        net.train()
        # training
        for i in range(len_dataloader_train):
            data_train = data_train_iter.__next__()
            fixed, moving, fixed_seg, moving_seg = data_train

            fixed = torch.unsqueeze(fixed, dim=1)
            moving = torch.unsqueeze(moving, dim=1)
            fixed_seg = torch.unsqueeze(fixed_seg, dim=1)
            moving_seg = torch.unsqueeze(moving_seg, dim=1)
            fixed_seg_onehot = mask_to_one_hot(fixed_seg, cls_num)
            moving_seg_onehot = mask_to_one_hot(moving_seg, cls_num)

            flow, warped = net(fixed, moving, 'train')

            err_img = image_loss_func(fixed, warped) * weight[0]
            err_flow = deformation_loss_func(flow) * weight[1]
            warped_seg_onehot = stn(moving_seg_onehot, flow)
            err_dice = compute_per_channel_dice(warped_seg_onehot, fixed_seg_onehot, cls_num) * weight[2]
            err = err_img + err_flow + err_dice

            epoch_sim_loss.append(err_img.item())
            epoch_grad_loss.append(err_flow.item())
            epoch_total_loss.append(err.item())

            # optimize
            optimizer.zero_grad()
            err.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        print("epoch: %d/%d, training time cost: %ds" % (epoch + 1, n_epoch, epoch_time))

        # validation
        dice_results = []
        net.eval()
        for i in range(len_dataloader_valid):
            data_valid = data_valid_iter.__next__()
            fixed, moving, fixed_seg, moving_seg = data_valid

            fixed = torch.unsqueeze(fixed, dim=1)
            moving = torch.unsqueeze(moving, dim=1)
            fixed_seg = torch.unsqueeze(fixed_seg, dim=1)
            moving_seg = torch.unsqueeze(moving_seg, dim=1)

            with torch.no_grad():
                flow, warped = net(fixed, moving, 'test')

            warped_seg = stn(moving_seg, flow, mode='nearest')[0, 0, :, :, :]
            dice_score_reg = dicegup(warped_seg, fixed_seg[0, 0, :, :, :])
            dice_score_reg = dice_score_reg.cpu().numpy()
            dice_results.append(dice_score_reg)

        dice = np.mean(dice_results)

        if dice > best_record:
            best = dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path + 'DGMIR_{%d}.pth' % epoch)
            print("BEST ckpt have saved, best Dice score: %.6e" % dice)

        print("total loss:%.4e, sim loss:%.4e, grad loss:%.4e" % (
            np.mean(epoch_total_loss),
            np.mean(epoch_sim_loss),
            np.mean(epoch_grad_loss),
        ))
