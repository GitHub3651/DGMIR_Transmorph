import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReplicationPad3d
from torch.autograd import Variable
import numpy as np
import math
import nibabel as nib


def compute_local_sums(I, J, filter, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filter, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filter, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filter, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filter, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filter, stride=stride, padding=padding)

    win_size = np.prod(win) * I.shape[1]
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def ncc_loss(I, J, win=None, mind=False):
    if win is None:
        win = [9] * 3

    sum_filter = torch.ones(1, I.shape[1], *win).to("cuda")
    pad = math.floor(win[0] / 2)
    stride = (1, 1, 1)
    pading = (pad, pad, pad)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filter, stride, pading, win)

    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def mask_to_one_hot(mask, n_classes):
    """
    Convert a segmentation mask to one-hot coded tensor
    :param mask: mask tensor of size Bx1xDxMxN
    :param n_classes: number of classes
    :return: one_hot: BxCxDxMxN
    """
    one_hot_shape = list(mask.shape)
    one_hot_shape[1] = n_classes
    mask_one_hot = torch.zeros(one_hot_shape).to(mask.device)
    mask_one_hot.scatter_(1, mask.long(), 1)
    return mask_one_hot


def compute_per_channel_dice(input, target, classes=27, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    # assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    if input.size() != target.size():
        target = mask_to_one_hot(target, n_classes=classes)

    # input = flatten(input)
    # target = flatten(target)
    # target = target.float()\

    input = input.contiguous().view(input.size()[1], -1)
    target = target.contiguous().view(target.size()[1], -1).float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    dice_score = 2 * (intersect / denominator.clamp(min=epsilon))
    return -dice_score.mean()


def dicegup(im1, atlas):
    unique_class = atlas.unique()
    ret = 0
    num_count = 0
    # im1 = torch.tensor(im1, dtype=torch.int64)
    # atlas = torch.tensor(atlas, dtype=torch.int64)
    im1 = im1.int()
    atlas = atlas.int()
    for i in unique_class:
        t1 = (im1 == i).sum()
        t2 = (atlas == i).sum()
        if i == 0 or t1 == 0 or t2 == 0:
            continue
        ret += (atlas[im1 == i] == i).sum() * 2.0 / (t1 + t2)
        num_count += 1

    if num_count == 0:
        return ret
    else:
        return ret / num_count



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        # y_true = (y_true - y_true.mean()) / y_true.std()
        # y_pred = (y_pred - y_pred.mean()) / y_pred.std()
        return torch.mean((y_true - y_pred) ** 2)


class Grad2:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class Grad(nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l2'):
        super(Grad, self).__init__()
        self.penalty = penalty

    def _diffs(self, y):  # y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        df = [None] * ndims
        for i in range(ndims):
            d = i + 2  # y shape(bs, c, d, h, w)
            # permute dimensions to put the ith dimension first
            #            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            #            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))

        return df

    def forward(self, pred):
        ndims = pred.ndimension() - 2
        if pred.is_cuda:
            df = Variable(torch.zeros(1).cuda())
        else:
            df = Variable(torch.zeros(1))
        for f in self._diffs(pred):
            if self.penalty == 'l1':
                df += f.abs().mean() / ndims
            else:
                assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df += f.pow(2).mean() / ndims
        return df


def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise squared Euclidean distance of input coordinates.

    Args:
        x: input coordinates, input shape should be (1, dim, #input points)
    Returns:
        dist: pairwise distance matrix, (#input points, #input points)
    """
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


class MINDSSCLoss(nn.Module):
    """
    Modality-Independent Neighbourhood Descriptor Dissimilarity Loss for Image Registration
    References: https://link.springer.com/chapter/10.1007/978-3-642-40811-3_24

    Args:
        radius (int): radius of self-similarity context.
        dilation (int): the dilation of neighbourhood patches.
        penalty (str): the penalty mode of mind dissimilarity loss.
    """

    def __init__(
            self,
            radius: int = 2,
            dilation: int = 2,
            penalty: str = 'l2',
    ) -> None:
        super().__init__()
        self.kernel_size = radius * 2 + 1
        self.dilation = dilation
        self.radius = radius
        self.penalty = penalty
        self.mshift1, self.mshift2, self.rpad1, self.rpad2 = self.build_kernels()

    def build_kernels(self):
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                                          [1, 1, 2], [2, 1, 1], [1, 2, 1]]).long()

        # squared distances
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask, square distance equals 2
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        # self-similarity context: 12 elements
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6,
                                                           1).view(-1,
                                                                   3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 +
                         idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift1.requires_grad = False

        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1,
                                                           1).view(-1,
                                                                   3)[mask, :]
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 +
                         idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        mshift2.requires_grad = False

        # maintain the output size
        rpad1 = ReplicationPad3d(self.dilation)
        rpad2 = ReplicationPad3d(self.radius)
        return mshift1, mshift2, rpad1, rpad2

    def mind(self, img: torch.Tensor) -> torch.Tensor:
        mshift1 = self.mshift1.to(img)
        mshift2 = self.mshift2.to(img)
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2(
            (F.conv3d(self.rpad1(img), mshift1, dilation=self.dilation) -
             F.conv3d(self.rpad1(img), mshift2, dilation=self.dilation)) ** 2),
            self.kernel_size,
            stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var,
                               mind_var.mean().item() * 0.001,
                               mind_var.mean().item() * 1000)
        mind = torch.div(mind, mind_var)
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:,
               torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(
               ), :, :, :]

        return mind

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute the MIND-SSC loss.

        Args:
            source: source image, tensor of shape [BNHWD].
            target: target image, tensor fo shape [BNHWD].
        """
        assert source.shape == target.shape, 'input and target must have the same shape.'
        if self.penalty == 'l1':
            mind_loss = torch.abs(self.mind(source) - self.mind(target))
        elif self.penalty == 'l2':
            mind_loss = torch.square(self.mind(source) - self.mind(target))
        else:
            raise ValueError(
                f'Unsupported penalty mode: {self.penalty}, available modes are l1 and l2.'
            )

        return torch.mean(mind_loss)  # the batch and channel average

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(radius={self.radius},'
                     f'dilation={self.dilation},'
                     f'penalty=\'{self.penalty}\')')
        return repr_str


class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0., self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)
