from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class SpatialTransformer2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self):
        super().__init__()
        # self.mode = mode

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)

    def forward(self, src, flow, mode='bilinear'):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # new locations
        new_locs = grid + flow
        # shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=mode)


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        # size_tensor = sample_grid.size()
        # sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
        # sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
        # sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def compute_module_memory_usage(model, input_size, device='cuda'):
    """
    计算 PyTorch 模型中每个模块的显存占用量（参数和激活值）。

    参数：
        - model: nn.Module，PyTorch 模型。
        - input_size: tuple，输入张量的形状（例如 (batch_size, channels, height, width)）。
        - device: str，设备类型（'cuda' 或 'cpu'）。

    返回：
        - memory_usage: dict，每个模块和总的显存占用量（单位：MB）。
    """
    model = model.to(device)
    memory_usage = {}

    # 将模型设置为评估模式，避免 Dropout 等影响
    model.eval()

    # 创建一个输入张量
    # dummy_input = torch.randn(*input_size).to(device)

    # 钩子函数，用于捕获每个模块的激活值大小
    def forward_hook(module, input, output):
        # 计算激活值的显存占用
        activation_memory = 0
        if isinstance(output, (tuple, list)):
            for out in output:
                activation_memory += out.numel() * out.element_size()
        else:
            activation_memory = output.numel() * output.element_size()

        # 记录显存占用（单位：MB）
        memory_usage[module] = activation_memory / (1024 ** 2)

    # 注册钩子到每个子模块
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(forward_hook))

    # 执行一次前向传播
    with torch.no_grad():
        model(input_size[0], input_size[1], input_size[2])

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 计算每个模块的参数显存占用
    for name, module in model.named_modules():
        param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
        param_memory /= (1024 ** 2)  # 转为 MB

        # 总显存占用 = 参数显存 + 激活值显存
        if module in memory_usage:
            memory_usage[module] += param_memory
        else:
            memory_usage[module] = param_memory

    # 计算模型的总显存占用量
    total_memory_usage = sum(memory_usage.values())

    # 返回每个模块的显存占用
    module_memory_usage = {}
    for name, module in model.named_modules():
        if module in memory_usage:
            module_memory_usage[name] = memory_usage[module]

    return module_memory_usage, total_memory_usage


def count_labels(nii_file_path):
    """
    统计 NIfTI 图像中的分割标签值及其频次。

    参数:
        nii_file_path (str): NIfTI 图像文件的路径。

    返回:
        dict: 一个字典，键为标签值，值为对应的频次。
    """
    # 加载 NIfTI 图像
    img = nib.load(nii_file_path)

    # 获取图像数据
    data = img.get_fdata()

    # 将数据转换为整数类型（如果是浮点类型）
    data = data.astype(np.int32)

    # 获取唯一标签值及其频次
    unique_labels, counts = np.unique(data, return_counts=True)

    # 将结果存储为字典
    label_count_dict = dict(zip(unique_labels, counts))

    return label_count_dict
