# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
class cal_sf_by_net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        with torch.no_grad():
            k_size=input.shape[3]//40
            output = self.fusion_channel_sf(input,
                                            kernel_radius=k_size)
            # todo 参数用多大要自己改 820*600的尺寸默认15,1640*1200的尺寸是30，以此类推
        return output

    @staticmethod
    # ref SESF-Fuse: an unsupervised deep model for multi-focus image fusion
    def fusion_channel_sf(f1, kernel_radius=5):  # default=5
        """
        Perform channel sf fusion two features
        """
        device = f1.device
        b, c, h, w = f1.shape  # 假设[1,80,5,5]
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
            .reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1).to(
            device)  # 卷积核weight.shape=[out_channels,in_channels,h,w]  这里c==out_channels
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
            .reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1).to(device)  # 需要输出维度个卷积核 所以要repeat成c个
        f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c).to(
            device)  # 每组计算被in_channels/groups个channels的卷积核计算
        f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c).to(
            device)  # 对输入tensor f1 进行卷积操作，卷积核为x_shift_kernel 且都用同一个卷积核计算
        f1_grad = torch.sqrt(torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)).to(
            device)  # RF^2+CF^2  RF与SF 两幅特征图相加的图
        kernel_size = kernel_radius * 2 + 1  # 2R+1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)  # [80,1,11,11]
        kernel_padding = kernel_size // 2  # padding==5
        f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1).to(device)
        f1_sf_np = f1_sf.squeeze()
        return f1_sf_np
