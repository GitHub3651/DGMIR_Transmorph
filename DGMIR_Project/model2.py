import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functions import SpatialTransformer2
from torch.distributions.normal import Normal


class DualPath_encoder(nn.Module):
    def __init__(self):
        super(DualPath_encoder, self).__init__()
        # Define layers
        self.Co_layer_1 = nn.Conv3d(1, 16, 3, stride=1, padding=1)
        self.Co_layer_2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.Co_layer_3 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.Co_layer_4 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, fixed, moving):
        # Fixed image path
        x_fix_1 = self.relu(self.Co_layer_1(fixed))
        x_fix_2 = self.relu(self.Co_layer_2(x_fix_1))
        x_fix_3 = self.relu(self.Co_layer_3(x_fix_2))
        x_fix_4 = self.relu(self.Co_layer_4(x_fix_3))

        # Moving image path
        x_mov_1 = self.relu(self.Co_layer_1(moving))
        x_mov_2 = self.relu(self.Co_layer_2(x_mov_1))
        x_mov_3 = self.relu(self.Co_layer_3(x_mov_2))
        x_mov_4 = self.relu(self.Co_layer_4(x_mov_3))
        return x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4


class Mix(nn.Module):
    def __init__(self, channel):
        super(Mix, self).__init__()
        self.w = torch.nn.Parameter(torch.ones(1, channel, 1, 1, 1), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor + fea2 * (1 - mix_factor)
        return out


class MFRG(nn.Module):
    def __init__(self, in_channels, reduction=8, pre_fea=False):
        super(MFRG, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.pre_fea = pre_fea

        if self.pre_fea:
            self.fc1 = nn.Conv3d(in_channels * 3, in_channels * 3 // reduction, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv3d(in_channels * 3 // reduction, in_channels * 3, 1, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.reduce_channels = nn.Conv3d(in_channels * 3, in_channels, 1, bias=False)
            self.mix = Mix(3 * in_channels)
        else:
            self.fc1 = nn.Conv3d(in_channels * 2, in_channels * 2 // reduction, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv3d(in_channels * 2 // reduction, in_channels * 2, 1, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.reduce_channels = nn.Conv3d(in_channels * 2, in_channels, 1, bias=False)
            self.mix = Mix(2 * in_channels)

    def forward(self, f, m, prefea):
        if self.pre_fea:
            input = torch.cat((f, m, prefea), dim=1)
            x1 = self.fc2(self.relu1(self.fc1(self.avg_pool(input)))).squeeze(-1).squeeze(-1)
            x2 = self.fc2(self.relu1(self.fc1(self.max_pool(input)))).squeeze(-1).squeeze(-1)
            mat = torch.matmul(x1, x2.transpose(-1, -2))

            out1 = torch.sum(mat, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (1,64,1,1,1)
            out1 = self.sigmoid(out1)
            out2 = torch.sum(mat, dim=2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out2 = self.sigmoid(out2)

            out = self.mix(out1, out2)
            attention = self.reduce_channels(out)
            f_s = attention * f
            m_s = attention * m
        else:
            input = torch.cat((f, m), dim=1)
            x1 = self.fc2(self.relu1(self.fc1(self.avg_pool(input)))).squeeze(-1).squeeze(-1)
            x2 = self.fc2(self.relu1(self.fc1(self.max_pool(input)))).squeeze(-1).squeeze(-1)
            mat = torch.matmul(x1, x2.transpose(-1, -2))

            out1 = torch.sum(mat, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (1,64,1,1,1)
            out1 = self.sigmoid(out1)
            out2 = torch.sum(mat, dim=2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out2 = self.sigmoid(out2)

            out = self.mix(out1, out2)
            attention = self.reduce_channels(out)
            f_s = attention * f
            m_s = attention * m
        return f_s, m_s


class MORG(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=2):
        super(MORG, self).__init__()
        self.conv_3 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv_block_s = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU()
        )
        self.conv_3_global = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        conv3_ker = torch.ones(channels, channels, 3, 3, 3, device='cuda') / 27.0
        self.conv_3_global.weight.data = conv3_ker

        self.conv_5_global = nn.Conv3d(channels, channels, kernel_size=5, stride=1, padding=2)
        conv5_ker = torch.ones(channels, channels, 5, 5, 5, device='cuda') / 25.0
        self.conv_5_global.weight.data = conv5_ker

        self.temp1 = nn.Parameter(torch.ones([1, channels, 1, 1, 1], dtype=torch.float32), requires_grad=True)
        self.temp2 = nn.Parameter(torch.ones([1, channels, 1, 1, 1], dtype=torch.float32), requires_grad=True)

        if padding == 2:
            self.conv_3_dila = nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=padding, groups=channels)
        elif padding == 3:
            self.conv_3_dila = nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=padding, groups=channels)
        self.conv_3_point_s = nn.Conv3d(channels, channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm3d(channels)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, y):
        b, c, h, w, d = x.size()
        x_1 = self.conv_3(x)
        x_1 = self.norm(x_1)
        x_1 = self.relu(x_1)
        x_2 = self.conv_3_dila(x)
        x_2 = self.norm(x_2)
        x_2 = self.relu(x_2)

        y_1 = self.conv_3(y)
        y_1 = self.norm(y_1)
        y_1 = self.relu(y_1)
        y_2 = self.conv_3_dila(y)
        y_2 = self.norm(y_2)
        y_2 = self.relu(y_2)

        spa_cor_1 = torch.cat([x_1, x_2], dim=1)
        spa_cor_1_avg = torch.cat((self.temp1 * self.conv_3_global(x_1), self.temp1 * self.conv_5_global(x_2)), dim=1)
        spa_cor_1 = self.conv_block_s(spa_cor_1 - spa_cor_1_avg)

        spa_cor_2 = torch.cat([y_1, y_2], dim=1)
        spa_cor_2_avg = torch.cat((self.temp2 * self.conv_3_global(y_1), self.temp2 * self.conv_5_global(y_2)), dim=1)
        spa_cor_2 = self.conv_block_s(spa_cor_2 - spa_cor_2_avg)

        return spa_cor_1, spa_cor_2


class RegHead_block(nn.Module):
    def __init__(self, in_channels: int, levels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.InstanceNorm3d(in_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(in_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, stride=1)
        )

        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
        self.levels = levels

    def forward(self, x_in):
        if self.levels >= 1:
            x_in = self.conv1(x_in)
            if self.levels >= 2:
                x_in = self.conv2(x_in)

        flow = self.reg_head(x_in)
        return flow


class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear', isimg=False):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode
        self.isimg = isimg

    def forward(self, x):
        if self.isimg:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        else:
            if self.factor < 1:
                # resize first to save memory
                x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
                x = self.factor * x

            elif self.factor > 1:
                # multiply first to save memory
                x = self.factor * x
                x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x


class Flow_decoder(nn.Module):
    def __init__(self):
        super(Flow_decoder, self).__init__()

        self.mfrg_64_1 = MFRG(in_channels=64, reduction=16, pre_fea=False)
        self.mfrg_64_2 = MFRG(in_channels=32, reduction=16, pre_fea=False)
        self.mfrg_32 = MFRG(in_channels=32, reduction=8, pre_fea=False)
        self.mfrg_16 = MFRG(in_channels=16, reduction=4, pre_fea=False)

        self.morg_2 = MORG(64, 3, 2)
        self.morg_3 = MORG(32, 3, 2)
        self.morg_4 = MORG(32, 3, 2)
        self.morg_5 = MORG(16, 3, 2)

        self.reg_head_4 = RegHead_block(64, 1, 32)
        self.reg_head_3 = RegHead_block(32, 1, 32)
        self.reg_head_2 = RegHead_block(32, 1, 16)
        self.reg_head_1 = RegHead_block(16, 1, 16)

        self.Resize_transformer_flow = ResizeTransformer_block(2, mode='trilinear')
        self.spatialTransformer = SpatialTransformer2()

    def forward(self, encoder_result):
        x_fix_1, x_mov_1, x_fix_2, x_mov_2, x_fix_3, x_mov_3, x_fix_4, x_mov_4 = encoder_result

        x_fix_4, x_mov_4 = self.mfrg_64_1(x_fix_4, x_mov_4, None)
        x_fix_4, x_mov_4 = self.morg_2(x_fix_4, x_mov_4)
        x = torch.cat([x_fix_4, x_mov_4], dim=1)
        flow = self.reg_head_4(x)

        flow = self.Resize_transformer_flow(flow)
        warped_mov_3 = self.spatialTransformer(x_mov_3, flow)
        x_fix_3, warped_mov_3 = self.mfrg_64_2(x_fix_3, warped_mov_3, None)
        x_fix_3, warped_mov_3 = self.morg_3(x_fix_3, warped_mov_3)
        x = torch.cat([x_fix_3, warped_mov_3], dim=1)
        flow_3 = self.reg_head_3(x)
        flow = self.spatialTransformer(flow, flow_3) + flow_3

        flow = self.Resize_transformer_flow(flow)
        warped_mov_2 = self.spatialTransformer(x_mov_2, flow)
        x_fix_2, warped_mov_2 = self.mfrg_32(x_fix_2, warped_mov_2, None)
        x_fix_2, warped_mov_2 = self.morg_4(x_fix_2, warped_mov_2)
        x = torch.cat([x_fix_2, warped_mov_2], dim=1)
        flow_2 = self.reg_head_2(x)
        flow = self.spatialTransformer(flow, flow_2) + flow_2

        flow = self.Resize_transformer_flow(flow)
        warped_mov_1 = self.spatialTransformer(x_mov_1, flow)
        x_fix_1, warped_mov_1 = self.mfrg_16(x_fix_1, warped_mov_1, None)
        x_fix_1, warped_mov_1 = self.morg_5(x_fix_1, warped_mov_1)
        x = torch.cat([x_fix_1, warped_mov_1], dim=1)
        flow_1 = self.reg_head_1(x)
        flow = self.spatialTransformer(flow, flow_1) + flow_1

        return flow


class DGMIR(nn.Module):
    def __init__(self, vol_size):
        super(DGMIR, self).__init__()
        self.vol_size = vol_size
        self.channels = [16, 32, 32, 64]
        self.encoder = DualPath_encoder()
        self.decoder = Flow_decoder()
        self.spatialTransformer = SpatialTransformer2()

    def forward(self, fixed, moving, mode):
        features = self.encoder(fixed, moving)
        flow = self.decoder(features)
        warped = self.spatialTransformer(moving, flow)

        if mode == 'train':
            return flow, warped
        if mode == 'test':
            return flow, warped
