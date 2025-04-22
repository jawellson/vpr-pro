import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from mvector.models.pooling import TemporalAveragePooling, TemporalStatsPool, AttentiveStatisticsPooling

__all__ = ['ERes2Net', 'ERes2NetV2']


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class EFC(nn.Module):   
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigmoid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1) * 0.1)  # 初始化标准差调小
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.conv4_global = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        
        # 使用 ModuleList 存储多个 interact 层
        self.interacts = nn.ModuleList([
            nn.Conv2d(c2 // 4, c2 // 4, kernel_size=1, stride=1) for _ in range(4)
        ])
    
    def forward(self, x1, x2):
        # 全局卷积操作
        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigmoid(bn_x).clamp(min=self.eps, max=1.0 - self.eps)  # Clamp 防止极端值
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigmoid(bn_x2).clamp(min=self.eps, max=1.0 - self.eps)  # Clamp 防止极端值
        X_GLOBAL = global_conv1 + global_conv2

        # 全局特征处理
        x_conv4 = self.conv4_global(X_GLOBAL)
        X_4_sigmoid = self.sigmoid(x_conv4).clamp(min=self.eps, max=1.0 - self.eps)  # Clamp 防止极端值
        X_ = X_4_sigmoid * X_GLOBAL
        X_chunks = X_.chunk(4, dim=1)
        out = []
        for group_id in range(4):
            interact = self.interacts[group_id]
            out_1 = interact(X_chunks[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.view(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True).clamp(min=self.eps)  # 防止除以零
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=2)
            x1 = x_2_2.view(N, C, H, W)
            x1 = X_chunks[group_id] * x1
            out.append(x1)
        out = torch.cat(out, dim=1)
        
        # 规范化操作
        N, C, H, W = out.size()
        x_add_1 = out.view(N, self.group_num, -1)
        x_shape_1 = X_GLOBAL.view(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True).clamp(min=self.eps)  # 防止除以零
        std_1 = x_shape_1.std(dim=2, keepdim=True).clamp(min=self.eps)    # 防止除以零
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.view(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta).clamp(min=-1e5, max=1e5)  # Clamp 防止数值过大
        
        # 权重计算
        weight_x3 = self.Apt(X_GLOBAL)
        reweights = self.sigmoid(weight_x3).clamp(min=self.eps, max=1.0 - self.eps)  # Clamp 防止极端值
        x_up_1 = (reweights >= weight_1).float()
        x_low_1 = (reweights < weight_1).float()
        x_up_2 = (reweights >= weight_2).float()
        x_low_2 = (reweights < weight_2).float()
        x_up = x_up_1 * X_GLOBAL + x_up_2 * X_GLOBAL
        x_low = x_low_1 * X_GLOBAL + x_low_2 * X_GLOBAL
        
        # 卷积操作
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_generator(x_low).clamp(min=self.eps, max=1.0 - self.eps)  # Clamp 防止极端值
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw
        xL = xL + x_gui
        xL = torch.nan_to_num(xL, nan=0.0, posinf=1e5, neginf=-1e5)  # 最后一步替换可能的 NaN
        return xL


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


class BasicBlockERes2Net(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale
        self.gam = GAM_Attention(in_channels = width * 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.gam(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(EFC(c1=width, c2=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)


        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2Net(nn.Module):
    def __init__(self,
                 input_size,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 mul_channel=1,
                 expansion=2,
                 base_width=32,
                 scale=2,
                 embd_dim=192,
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.expansion = expansion
        self.feat_dim = input_size
        self.embd_dim = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1,
                                       base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2,
                                       base_width=base_width, scale=scale)


        # Downsampling module for each layer
        self.layer1_downsample = nn.Conv2d(m_channels * 2 * mul_channel, m_channels * 4 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4 * mul_channel, m_channels * 8 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8 * mul_channel, m_channels * 16 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.fuse_mode12 = EFC(c1=m_channels * 4 * mul_channel, c2=m_channels * 4 * mul_channel)
        self.fuse_mode123 = EFC(c1=m_channels * 8 * mul_channel, c2=m_channels * 8 * mul_channel)
        self.fuse_mode1234 = EFC(c1=m_channels * 16 * mul_channel, c2=m_channels * 16 * mul_channel)


        self.gam = GAM_Attention(in_channels = 512)

        self.n_stats = 2
        self.pooling = TemporalStatsPool()
        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.expansion, self.in_planes, planes, stride, base_width, scale))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        # print(fuse_out1234.shape) 512
        fuse_out1234 = self.gam(fuse_out1234)

        stats = self.pooling(fuse_out1234)


        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a
-*- coding: utf-8 -*-
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Assume these pooling layers are defined elsewhere or imported correctly
# # from mvector.models.pooling import TemporalAveragePooling, TemporalStatsPool, AttentiveStatisticsPooling
# # Placeholder for TemporalStatsPool if not available externally for running this script directly
# class TemporalStatsPool(nn.Module):
#     """ Aggregates statistics over the time dimension (Placeholder if not imported) """
#     def __init__(self):
#         super(TemporalStatsPool, self).__init__()

#     def forward(self, x):
#         # Ensure tensor has spatial dimensions
#         if x.dim() < 4 or x.shape[2] == 0 or x.shape[3] == 0:
#              print(f"Warning: TemporalStatsPool received input with shape {x.shape}. Returning zeros.")
#              # Calculate expected output dimension based on channels
#              # Assuming output is mean and std concatenated
#              b, c = x.shape[0], x.shape[1]
#              return torch.zeros(b, c * 2, device=x.device)

#         b, c, h, w = x.shape
#         # Reshape for pooling: Combine channel and frequency dims, pool over time
#         # Original assumes pooling over last dim (W) after view(b, c*h, w)
#         # Let's adapt assuming standard spectrogram (B, C, Freq, Time) -> Pool over Time (dim 3)
#         epsilon = 1e-10
#         mean = x.mean(dim=3) # Mean over time
#         std = x.std(dim=3).clamp(min=epsilon) # Std over time
#         # Combine Freq stats: Flatten mean/std across Freq dim
#         stats = torch.cat((mean.view(b, -1), std.view(b, -1)), dim=1)
#         # Expected output: (B, 2 * C * H)
#         return stats

# # --- Import torchsummary and thop ---
# try:
#     from torchsummary import summary as torchsummary_summary
#     _torchsummary_available = True
#     print("torchsummary 库已找到。")
# except ImportError:
#     print("警告：未找到 torchsummary 库。将无法打印详细模型信息。")
#     print("请运行 'pip install torchsummary' 来安装。")
#     _torchsummary_available = False

# try:
#     from thop import profile
#     _thop_available = True
#     print("thop 库已找到。")
# except ImportError:
#     print("警告：未找到 thop 库。将无法计算 FLOPs。")
#     print("请运行 'pip install thop' 来安装。")
#     _thop_available = False

# # --- Depthwise Separable Convolution Module ---
# class DepthwiseSeparableConv(nn.Module):
#     """
#     深度可分离卷积模块：Depthwise Conv -> Pointwise Conv
#     不包含内部的BN和ReLU，以便灵活替换标准卷积层。
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
#         super().__init__()
#         # Ensure channels are at least 1 to avoid errors with empty tensors/groups
#         in_channels = max(1, in_channels)
#         out_channels = max(1, out_channels)
#         # Depthwise Convolution: groups = in_channels
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
#                                    stride=stride, padding=padding,
#                                    groups=in_channels, bias=bias)
#         # Pointwise Convolution: 1x1 kernel
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                    stride=1, padding=0, bias=bias)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# # --- Modified Convolution Definitions ---
# def conv1x1(in_planes, out_planes, stride=1):
#     "1x1 convolution (Pointwise) without padding"
#     # Ensure channels are at least 1
#     in_planes = max(1, in_planes)
#     out_planes = max(1, out_planes)
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 depthwise separable convolution with padding"
#     # Ensure channels are at least 1
#     in_planes = max(1, in_planes)
#     out_planes = max(1, out_planes)
#     # Use DepthwiseSeparableConv instead of standard Conv2d
#     return DepthwiseSeparableConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# # --- Other Class Definitions ---
# class ReLU(nn.Hardtanh):
#     def __init__(self, inplace=False):
#         super(ReLU, self).__init__(0, 20, inplace)

#     def __repr__(self):
#         inplace_str = 'inplace' if self.inplace else ''
#         return self.__class__.__name__ + ' (' + inplace_str + ')'

# # --- Modified EFC using DSConv for interaction ---
# class EFC(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         c1 = max(c1, 1); c2 = max(c2, 1) # Ensure channels >= 1
#         # Use a reasonable default, avoid division by zero if c2 is small
#         num_groups_default = 4
#         # Ensure group_num is at least 1 and c2 is divisible by group_num if possible
#         if c2 > 0 and c2 // num_groups_default > 0:
#             self.group_num = min(num_groups_default, c2 // (c2 // num_groups_default))
#         else:
#             self.group_num = 1 # Fallback to 1 group if c2 is too small or zero

#         self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
#         self.bn_conv1 = nn.BatchNorm2d(c2)
#         self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)
#         self.bn_conv2 = nn.BatchNorm2d(c2)
#         self.conv4_global = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False) # Kept global conv as 1x1 standard

#         self.sigmoid = nn.Sigmoid()
#         self.eps = 1e-10

#         # Initialize gamma/beta similar to the first script
#         self.gamma = nn.Parameter(torch.ones(c2, 1, 1)) # Initialize gamma to 1
#         self.beta = nn.Parameter(torch.zeros(c2, 1, 1)) # Initialize beta to 0

#         # Gate generator similar to the first script
#         self.gate_generator = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(c2, max(c2 // 4, 1), 1, 1, bias=False), # Ensure intermediate channel >= 1
#             nn.ReLU(True),
#             nn.Conv2d(max(c2 // 4, 1), c2, 1, 1, bias=False),
#             nn.Sigmoid() # Use Sigmoid like the first script's gate
#         )

#         # DWConv part similar to the first script
#         self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False)
#         self.bn_dwconv = nn.BatchNorm2d(c2)
#         self.relu_dwconv = ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False) # Pointwise conv
#         self.bn_conv3 = nn.BatchNorm2d(c2)
#         self.Apt = nn.AdaptiveAvgPool2d(1)
#         self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False) # Pointwise conv

#         # Interaction part using DepthwiseSeparableConv
#         interact_channels = max(c2 // self.group_num, 1) if self.group_num > 0 else c2
#         self.interacts = nn.ModuleList([
#             DepthwiseSeparableConv(interact_channels, interact_channels, kernel_size=3, stride=1, padding=1, bias=False)
#             for _ in range(self.group_num)
#         ]) if self.group_num > 0 and interact_channels > 0 else nn.ModuleList() # Handle zero channels case

#         self.bn_interact = nn.ModuleList([
#              nn.BatchNorm2d(interact_channels) for _ in range(self.group_num)
#         ]) if self.group_num > 0 and interact_channels > 0 else nn.ModuleList()
#         self.relu_interact = ReLU(inplace=True)

#     def forward(self, x1, x2):
#         # Initial convolutions and fusion
#         global_conv1 = self.conv1(x1); bn_x1 = self.bn_conv1(global_conv1)
#         global_conv2 = self.conv2(x2); bn_x2 = self.bn_conv2(global_conv2)
#         X_GLOBAL = bn_x1 + bn_x2 # Fuse features
#         x_conv4_global = self.conv4_global(X_GLOBAL) # Apply global conv

#         # Grouped Interaction with DSConv
#         # Check if grouping is possible and modules exist
#         if self.group_num > 0 and len(self.interacts) > 0 and X_GLOBAL.size(1) > 0 and X_GLOBAL.size(1) % self.group_num == 0:
#             interact_channels = X_GLOBAL.size(1) // self.group_num
#             X_chunks = X_GLOBAL.chunk(self.group_num, dim=1)
#             out_interact = []
#             for group_id in range(self.group_num):
#                 chunk = X_chunks[group_id]
#                 out_chunk = self.interacts[group_id](chunk)
#                 out_chunk = self.bn_interact[group_id](out_chunk)
#                 out_chunk = self.relu_interact(out_chunk)
#                 out_interact.append(out_chunk)
#             out_interact_cat = torch.cat(out_interact, dim=1)
#         else: # Skip interaction if not possible
#             out_interact_cat = X_GLOBAL

#         # Group Normalization part
#         N, C, H, W = out_interact_cat.size()
#         # Check if group norm is applicable
#         if self.group_num > 0 and C > 0 and C % self.group_num == 0:
#              x_group = out_interact_cat.view(N, self.group_num, C // self.group_num, H, W)
#              mean = x_group.mean(dim=(2, 3, 4), keepdim=True)
#              std = x_group.std(dim=(2, 3, 4), keepdim=True).clamp(min=self.eps) # Clamp std
#              x_norm = (x_group - mean) / std
#              x_norm = x_norm.view(N, C, H, W) # Reshape back
#              # Apply learnable scale/shift (gamma/beta) if dimensions match
#              if self.gamma.shape[0] == C:
#                  x_gui = x_norm * self.gamma + self.beta
#              else: # Fallback if gamma/beta dims don't match (shouldn't happen with correct init)
#                  x_gui = x_norm
#         else: # Skip group norm if not applicable
#             x_gui = out_interact_cat

#         # Gating and splitting part
#         reweights = self.sigmoid(self.Apt(X_GLOBAL)) # Channel weights based on global avg pool
#         x_up = X_GLOBAL * reweights          # Weighted features (upper path)
#         x_low = X_GLOBAL * (1 - reweights)   # Weighted features (lower path)

#         # Lower path processing (DWConv)
#         x_low_dw = self.dwconv(x_low)
#         x_low_dw = self.bn_dwconv(x_low_dw)
#         x_low_dw = self.relu_dwconv(x_low_dw)
#         x_low_pw = self.conv3(x_low_dw) # Pointwise conv
#         x_low_pw = self.bn_conv3(x_low_pw)
#         gate = self.gate_generator(x_low) # Generate gate based on lower path input
#         x_low_gated = x_low_pw * gate       # Apply gate

#         # Upper path processing (Pointwise Conv)
#         x_up_pw = self.conv4(x_up)

#         # Combine paths and add normalized features
#         xL = x_low_gated + x_up_pw + x_gui
#         xL = torch.nan_to_num(xL, nan=0.0, posinf=1e5, neginf=-1e5) # Handle potential NaN/Inf
#         return xL

# # --- Modified GAM_Attention using DSConv ---
# class GAM_Attention(nn.Module):
#     def __init__(self, in_channels, rate=4):
#         super(GAM_Attention, self).__init__()
#         # Ensure channels are valid
#         in_channels = max(in_channels, 1)
#         inter_channels = max(in_channels // rate, 1) # Ensure intermediate channel >= 1

#         # Channel Attention (MLP) - Remains the same
#         self.channel_attention = nn.Sequential(
#             nn.Linear(in_channels, inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(inter_channels, in_channels)
#         )

#         # Spatial Attention - Use DepthwiseSeparableConv
#         self.spatial_attention = nn.Sequential(
#             # Replace standard Conv2d with DepthwiseSeparableConv
#             DepthwiseSeparableConv(in_channels, inter_channels, kernel_size=7, padding=3, bias=False),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             # Replace standard Conv2d with DepthwiseSeparableConv
#             DepthwiseSeparableConv(inter_channels, in_channels, kernel_size=7, padding=3, bias=False),
#             nn.BatchNorm2d(in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, h, w = x.shape
#         # Handle case with zero channels
#         if c == 0:
#             return x

#         # Channel Attention Path
#         # Use AdaptiveAvgPool2d for global descriptor, more standard than permute/view/linear
#         channel_descriptor = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)

#         # Check for zero descriptor to avoid NaN in sigmoid/attention
#         if torch.abs(channel_descriptor).sum() < 1e-9:
#              # If descriptor is near zero, use uniform attention weights
#              channel_att_weight = torch.ones(b, c, 1, 1, device=x.device)
#         else:
#             channel_att_raw = self.channel_attention(channel_descriptor).view(b, c, 1, 1)
#             channel_att_weight = self.sigmoid(channel_att_raw)

#         x_channel_att = x * channel_att_weight # Apply channel attention

#         # Spatial Attention Path (applied to channel-attended features)
#         spatial_att_raw = self.spatial_attention(x_channel_att)
#         spatial_att_weight = self.sigmoid(spatial_att_raw)

#         out = x_channel_att * spatial_att_weight # Apply spatial attention
#         return out

# # --- Renamed Res2Net Block using DSConv (via conv3x3) ---
# class Res2NetBlock(nn.Module):
#     # Renamed from BasicBlockERes2Net
#     def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
#         super(Res2NetBlock, self).__init__()
#         self.expansion = expansion
#         # Ensure width calculation is valid
#         width = max(int(math.floor(planes * (base_width / 64.0))), 1)
#         # Ensure channel counts are valid
#         out_c_conv1 = max(width * scale, 1)
#         in_planes = max(in_planes, 1)

#         self.conv1 = conv1x1(in_planes, out_c_conv1, stride) # 1x1 pointwise
#         self.bn1 = nn.BatchNorm2d(out_c_conv1)
#         self.relu = ReLU(inplace=True)
#         self.nums = max(scale, 1) # Ensure scale >= 1
#         self.width = width

#         convs = []; bns = []
#         # Create conv/bn pairs for each scale
#         for _ in range(self.nums):
#             # Use the modified conv3x3 which employs DepthwiseSeparableConv
#             convs.append(conv3x3(self.width, self.width))
#             bns.append(nn.BatchNorm2d(self.width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)

#         # Ensure output channels are valid
#         out_c_conv3 = max(planes * self.expansion, 1)
#         in_c_conv3 = max(self.width * self.nums, 1) # Input to final 1x1 depends on width*scale

#         self.conv3 = conv1x1(in_c_conv3, out_c_conv3) # Final 1x1 pointwise
#         self.bn3 = nn.BatchNorm2d(out_c_conv3)

#         # Shortcut connection
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != out_c_conv3:
#             self.shortcut = nn.Sequential(
#                 # Use standard Conv2d for shortcut projection if needed
#                 nn.Conv2d(in_planes, out_c_conv3, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_c_conv3)
#             )
#         # Removed GAM_Attention from inside the block, apply it later in the main model if needed

#     def forward(self, x):
#         shortcut = self.shortcut(x)
#         out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

#         # Handle cases where splitting isn't possible or scale is 0/1
#         if self.nums <= 0 or self.width <= 0 or out.size(1) < self.width * self.nums:
#             # If split is not possible, process the whole tensor with the first conv if available
#             if self.nums > 0 and len(self.convs) > 0:
#                  sp = self.convs[0](out); sp = self.bns[0](sp); processed_out = self.relu(sp)
#             else: # If no convs defined (e.g., scale=0), pass through
#                 processed_out = out
#             y = [processed_out] # Store the single processed output
#         else:
#             # Standard Res2Net split and process logic
#             spx = torch.split(out, self.width, 1)
#             y = [] # Store outputs of each scale path
#             for i in range(self.nums):
#                 if i == 0:
#                     sp = spx[i] # First split part
#                 else:
#                     # Add output of previous scale to current input split part
#                     # Check size compatibility before adding
#                     if y and y[-1].size() == spx[i].size():
#                         sp = y[-1] + spx[i]
#                     else: # If sizes mismatch (shouldn't happen with correct setup), use current split only
#                         sp = spx[i]

#                 # Apply DSConv (via conv3x3), BN, ReLU
#                 sp = self.convs[i](sp); sp = self.bns[i](sp); sp = self.relu(sp)
#                 y.append(sp) # Store result

#         # Concatenate results from different scales if available
#         if not y: # If y is empty (e.g., error case or scale=0)
#              out = torch.tensor([], device=x.device, dtype=x.dtype) # Create empty tensor
#         else:
#             out = torch.cat(y, 1)

#         # Apply final 1x1 conv if concatenated output is not empty and channels match
#         if out.numel() > 0 and out.size(1) == self.conv3.in_channels:
#              out = self.conv3(out)
#              out = self.bn3(out)
#              # Add shortcut
#              if out.shape == shortcut.shape:
#                  out = out + shortcut
#              else:
#                  print(f"Warning: Res2NetBlock output shape {out.shape} != shortcut shape {shortcut.shape}. Skipping residual connection.")
#                  # Potentially resize shortcut or output if mismatch is expected and fixable
#                  # For now, just skip the addition if shapes don't match
#         else: # If concatenation failed or channels mismatch, use shortcut directly
#             out = shortcut

#         out = self.relu(out) # Final ReLU
#         return out

# # --- Renamed EFC Fusion Block using DSConv (via conv3x3) and modified EFC ---
# class EFCFusionRes2NetBlock(nn.Module):
#     # Renamed from BasicBlockERes2Net_diff_AFF
#     def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
#         super(EFCFusionRes2NetBlock, self).__init__()
#         self.expansion = expansion
#         # Ensure width calculation is valid
#         width = max(int(math.floor(planes * (base_width / 64.0))), 1)
#         # Ensure channel counts are valid
#         out_c_conv1 = max(width * scale, 1)
#         in_planes = max(in_planes, 1)
#         planes = max(planes, 1) # Ensure output base planes >= 1

#         self.conv1 = conv1x1(in_planes, out_c_conv1, stride) # 1x1 pointwise
#         self.bn1 = nn.BatchNorm2d(out_c_conv1)
#         self.relu = ReLU(inplace=True)
#         self.nums = max(scale, 1) # Ensure scale >= 1
#         self.width = width

#         convs = []; bns = []
#         # Create conv/bn pairs for each scale
#         for i in range(self.nums):
#             # Use the modified conv3x3 which employs DepthwiseSeparableConv
#             convs.append(conv3x3(self.width, self.width))
#             bns.append(nn.BatchNorm2d(self.width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)

#         # Create EFC fusion modules between scales (nums-1 modules needed)
#         fuse_models = []
#         for _ in range(self.nums - 1):
#             # Only add EFC if width > 0, otherwise use Identity
#             if self.width > 0:
#                 # Use the modified EFC class
#                 fuse_models.append(EFC(c1=self.width, c2=self.width))
#             else:
#                 fuse_models.append(nn.Identity()) # Placeholder if width is zero
#         self.fuse_models = nn.ModuleList(fuse_models)

#         # Ensure output channels are valid
#         out_c_conv3 = max(planes * self.expansion, 1)
#         in_c_conv3 = max(self.width * self.nums, 1) # Input to final 1x1

#         self.conv3 = conv1x1(in_c_conv3, out_c_conv3) # Final 1x1 pointwise
#         self.bn3 = nn.BatchNorm2d(out_c_conv3)

#         # Shortcut connection
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != out_c_conv3:
#             self.shortcut = nn.Sequential(
#                 # Use standard Conv2d for shortcut projection
#                 nn.Conv2d(in_planes, out_c_conv3, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_c_conv3)
#             )

#     def forward(self, x):
#         shortcut = self.shortcut(x)
#         out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

#         # Handle cases where splitting isn't possible or scale is 0/1
#         if self.nums <= 0 or self.width <= 0 or out.size(1) < self.width * self.nums:
#              # If split not possible, process with first conv if available
#             if self.nums > 0 and len(self.convs) > 0:
#                  sp = self.convs[0](out); sp = self.bns[0](sp); processed_out = self.relu(sp)
#             else: # Pass through if no convs
#                 processed_out = out
#             y = [processed_out]
#         else:
#             # Res2Net split with EFC fusion logic
#             spx = torch.split(out, self.width, 1)
#             y = [] # Store results of each scale path
#             last_processed = None # Store the output of the previous scale for EFC input
#             for i in range(self.nums):
#                 current_input = spx[i]
#                 # Apply EFC fusion from the second scale onwards
#                 if i > 0 and i-1 < len(self.fuse_models) and last_processed is not None:
#                      # Check if EFC module exists and shapes match for fusion
#                     if isinstance(self.fuse_models[i - 1], EFC) and \
#                        last_processed.size(1) == self.width and \
#                        current_input.size(1) == self.width:
#                         sp = self.fuse_models[i - 1](last_processed, current_input) # Fuse previous output and current split
#                     else: # If EFC not applicable or shapes mismatch, use current split only
#                         sp = current_input
#                 else: # First scale, no fusion yet
#                     sp = current_input

#                 # Apply DSConv (via conv3x3), BN, ReLU
#                 sp = self.convs[i](sp); sp = self.bns[i](sp); sp = self.relu(sp)
#                 y.append(sp)
#                 last_processed = sp # Update last processed output

#         # Concatenate results if available
#         if not y:
#              out = torch.tensor([], device=x.device, dtype=x.dtype)
#         else:
#             out = torch.cat(y, 1)

#         # Apply final 1x1 conv if concatenated output is valid
#         if out.numel() > 0 and out.size(1) == self.conv3.in_channels:
#             out = self.conv3(out)
#             out = self.bn3(out)
#             # Add shortcut
#             if out.shape == shortcut.shape:
#                 out = out + shortcut
#             else:
#                  print(f"Warning: EFCFusionRes2NetBlock output shape {out.shape} != shortcut shape {shortcut.shape}. Skipping residual connection.")
#                  # Skip addition if shapes mismatch
#         else: # Use shortcut if concat failed or channels mismatch
#             out = shortcut

#         out = self.relu(out) # Final ReLU
#         return out

# # --- Main Model: ERes2Net_DSConv (Modified ERes2Net) ---
# class ERes2Net_DSConv(nn.Module):
#     # Renamed from ERes2Net to indicate DSConv usage
#     def __init__(self,
#                  input_size,
#                  # Use renamed blocks with DSConv as defaults
#                  block=Res2NetBlock,
#                  block_fuse=EFCFusionRes2NetBlock,
#                  num_blocks=[3, 4, 6, 3], # Default block counts
#                  m_channels=32,           # Base channel size
#                  # mul_channel=1,        # Removed, expansion handles channel increase
#                  expansion=2,             # Channel expansion factor in blocks
#                  base_width=32,           # Base width for Res2Net calculation
#                  scale=2,                 # Scale factor for Res2Net blocks
#                  embd_dim=192,            # Output embedding dimension
#                  two_emb_layer=False,     # Whether to use a second linear layer for embedding
#                  pooling_type='stats'):    # Added pooling type selection
#         super(ERes2Net_DSConv, self).__init__()

#         # Ensure valid parameters
#         input_size = max(input_size, 1)
#         m_channels = max(m_channels, 1)
#         embd_dim = max(embd_dim, 1)

#         self.in_planes = m_channels # Track current channel depth for _make_layer
#         self.expansion = expansion
#         self.embd_dim = embd_dim
#         self.two_emb_layer = two_emb_layer

#         # Initial Convolution: Use DepthwiseSeparableConv
#         self.conv1 = DepthwiseSeparableConv(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(m_channels)
#         self.relu1 = ReLU(inplace=True) # Use the custom ReLU

#         # --- Backbone Layers ---
#         layer_out_channels = [] # Store output channels of each main layer
#         # Layer 1: Uses 'block' (Res2NetBlock by default)
#         self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, base_width=base_width, scale=scale)
#         layer_out_channels.append(self.in_planes) # self.in_planes updated by _make_layer

#         # Layer 2: Uses 'block', increases channels, stride=2
#         self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, base_width=base_width, scale=scale)
#         layer_out_channels.append(self.in_planes)

#         # Layer 3: Uses 'block_fuse' (EFCFusionRes2NetBlock by default), increases channels, stride=2
#         self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, base_width=base_width, scale=scale)
#         layer_out_channels.append(self.in_planes)

#         # Layer 4: Uses 'block_fuse', increases channels, stride=2
#         self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, base_width=base_width, scale=scale)
#         layer_out_channels.append(self.in_planes)

#         l1_out_c, l2_out_c, l3_out_c, l4_out_c = layer_out_channels
#         # Expected channel sizes after layers (planes * expansion)
#         # l1_out_c = m_channels * expansion
#         # l2_out_c = m_channels * 2 * expansion
#         # l3_out_c = m_channels * 4 * expansion
#         # l4_out_c = m_channels * 8 * expansion

#         # --- Downsampling and Fusion Layers (using DSConv and modified EFC) ---
#         # Downsample Layer 1 output to match Layer 2 channel size for fusion
#         ds1_out_c = max(l2_out_c, 1) # Target channels = layer 2 output channels
#         self.layer1_downsample = nn.Sequential(
#             DepthwiseSeparableConv(max(l1_out_c, 1), ds1_out_c, kernel_size=3, padding=1, stride=2, bias=False),
#             nn.BatchNorm2d(ds1_out_c), ReLU(inplace=True)
#         ) if l1_out_c > 0 else nn.Identity()
#         self.fuse_mode12 = EFC(c1=ds1_out_c, c2=ds1_out_c) if ds1_out_c > 0 else nn.Identity()

#         # Downsample fused Layer 1&2 output to match Layer 3 channel size
#         ds2_out_c = max(l3_out_c, 1) # Target channels = layer 3 output channels
#         # Input to this downsampler is the output of fuse_mode12 (ds1_out_c channels)
#         self.layer2_downsample = nn.Sequential(
#             DepthwiseSeparableConv(ds1_out_c, ds2_out_c, kernel_size=3, padding=1, stride=2, bias=False),
#             nn.BatchNorm2d(ds2_out_c), ReLU(inplace=True)
#         ) if ds1_out_c > 0 else nn.Identity()
#         self.fuse_mode123 = EFC(c1=ds2_out_c, c2=ds2_out_c) if ds2_out_c > 0 else nn.Identity()

#         # Downsample fused Layer 1&2&3 output to match Layer 4 channel size
#         ds3_out_c = max(l4_out_c, 1) # Target channels = layer 4 output channels
#         # Input to this downsampler is the output of fuse_mode123 (ds2_out_c channels)
#         self.layer3_downsample = nn.Sequential(
#             DepthwiseSeparableConv(ds2_out_c, ds3_out_c, kernel_size=3, padding=1, stride=2, bias=False),
#             nn.BatchNorm2d(ds3_out_c), ReLU(inplace=True)
#         ) if ds2_out_c > 0 else nn.Identity()
#         self.fuse_mode1234 = EFC(c1=ds3_out_c, c2=ds3_out_c) if ds3_out_c > 0 else nn.Identity()

#         # Final Attention Module (using modified GAM_Attention with DSConv)
#         # Applied to the output of the final fusion stage (ds3_out_c channels)
#         self.gam = GAM_Attention(in_channels=ds3_out_c) if ds3_out_c > 0 else nn.Identity()
#         final_feature_channels = ds3_out_c # Channels going into pooling

#         # --- Pooling Layer ---
#         self.pooling_type = pooling_type
#         if pooling_type == 'stats':
#             self.pooling = TemporalStatsPool()
#             # Calculate feature dimension after StatsPooling (mean + std)
#             # Need the frequency dimension after all convolutions/downsampling
#             # Input: (B, 1, F, T) -> Conv1(s1) -> Layer1(s1) -> Layer2(s2) -> Layer3(s2) -> Layer4(s2)
#             # Strides: 1 (conv1) * 1 (L1) * 2 (L2) * 2 (L3) * 2 (L4) = 8
#             # Freq dim = ceil(input_size / 8)
#             freq_dim_after_conv = math.ceil(input_size / 8.0)
#             # Stats dim = 2 * channels * freq_dim
#             self.stats_dim_actual = max(int(2 * final_feature_channels * freq_dim_after_conv), 1)
#         elif pooling_type == 'avg':
#              self.pooling = TemporalAveragePooling() # Assumed to exist
#              # Avg pooling output dim = final_feature_channels
#              self.stats_dim_actual = max(final_feature_channels, 1)
#         elif pooling_type == 'attentive':
#              self.pooling = AttentiveStatisticsPooling(final_feature_channels, 128) # Assumed to exist, hidden_size=128 example
#              # Attentive pooling output dim = 2 * final_feature_channels
#              self.stats_dim_actual = max(2 * final_feature_channels, 1)
#         else:
#             raise ValueError(f"Unsupported pooling type: {pooling_type}")


#         # --- Embedding Layers ---
#         self.seg_1 = nn.Linear(self.stats_dim_actual, embd_dim)
#         if self.two_emb_layer:
#             self.relu_emb = nn.ReLU(inplace=True) # Standard ReLU for embedding
#             self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False) # BN before second linear
#             self.seg_2 = nn.Linear(embd_dim, embd_dim)
#         else:
#             # Use Identity if only one embedding layer
#             self.relu_emb = nn.Identity()
#             self.seg_bn_1 = nn.Identity()
#             self.seg_2 = nn.Identity()

#         # Initialize weights like the first script
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 try: # Kaiming for ReLU non-linearity
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 except ValueError: # Fallback for layers where ReLU might not be assumed (e.g., pointwise in DSConv sometimes)
#                     nn.init.normal_(m.weight, 0, 0.01) # Smaller init might be safer here
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                  # Check if affine transformation is enabled
#                  if m.affine:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight) # Xavier for linear layers
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
#         # This method now receives the renamed block classes (Res2NetBlock or EFCFusionRes2NetBlock)
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         current_planes = max(planes, 1) # Ensure target planes >= 1

#         # Create the first block which might have a stride != 1
#         try:
#             # Instantiate the passed block class (e.g., Res2NetBlock)
#             # self.in_planes is the number of input channels to this block
#             layers.append(block(self.expansion, self.in_planes, current_planes, stride=strides[0], base_width=base_width, scale=scale))
#             # Update self.in_planes for the next block in this layer
#             # Output channels of a block = planes * expansion
#             self.in_planes = current_planes * self.expansion
#         except Exception as e:
#             print(f"Error creating first block in layer: {e}")
#             layers.append(nn.Identity()) # Add identity on error

#         # Create remaining blocks in the layer (stride=1)
#         for i in range(1, num_blocks):
#             try:
#                 # Ensure in_planes is valid before creating block
#                 if self.in_planes > 0:
#                     # Instantiate the passed block class
#                     layers.append(block(self.expansion, self.in_planes, current_planes, stride=strides[i], base_width=base_width, scale=scale))
#                     # self.in_planes remains planes * expansion for subsequent blocks in the same layer
#                 else:
#                     layers.append(nn.Identity())
#             except Exception as e:
#                 print(f"Error creating block {i} in layer: {e}")
#                 layers.append(nn.Identity()) # Add identity on error

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # Input shape check
#         if x.dim() == 3:
#             # Assume (B, T, F) -> Permute to (B, F, T)
#              x = x.permute(0, 2, 1)
#         elif x.dim() == 4:
#              # Assume (B, 1, F, T) - Keep as is
#              pass
#         else:
#              raise ValueError(f"Expected 3D (B, T, F) or 4D (B, 1, F, T) input, got {x.dim()}D with shape {x.shape}")

#         B, C, F, T = x.shape # Get dimensions after potential permutation/check
#         if F == 0 or T == 0:
#             print(f"Warning: Zero dimension detected in input F={F}, T={T}. Returning zero embedding.")
#             return torch.zeros(B, self.embd_dim, device=x.device, dtype=x.dtype)

#         # Ensure input has 1 channel if needed (e.g., if input was B,F,T)
#         if C != 1:
#              x = x.unsqueeze(1) # Add channel dim: (B, F, T) -> (B, 1, F, T)

#         # --- Forward Pass ---
#         # 1. Initial DSConv + BN + ReLU
#         out = self.conv1(x); out = self.bn1(out); out = self.relu1(out)

#         # 2. Backbone Layers
#         out1 = self.layer1(out)    # Output channels: m_channels * expansion
#         out2 = self.layer2(out1)   # Output channels: m_channels*2 * expansion, Spatial dim halved
#         out3 = self.layer3(out2)   # Output channels: m_channels*4 * expansion, Spatial dim halved
#         out4 = self.layer4(out3)   # Output channels: m_channels*8 * expansion, Spatial dim halved

#         # 3. Fusion Path (Downsample + EFC)
#         # Fuse 1 and 2
#         out1_ds = self.layer1_downsample(out1) # Downsample out1
#         # Check shapes before EFC fusion
#         if isinstance(self.fuse_mode12, EFC) and out2.shape == out1_ds.shape:
#             fuse_out12 = self.fuse_mode12(out2, out1_ds)
#         else: # If shapes mismatch or EFC is Identity, use out2 directly
#             if not isinstance(self.fuse_mode12, EFC): fuse_out12 = out2
#             else:
#                  print(f"Warning: Shape mismatch for fuse_mode12: out2 {out2.shape}, out1_ds {out1_ds.shape}. Using out2.")
#                  fuse_out12 = out2


#         # Fuse 1&2 result with 3
#         fuse_out12_ds = self.layer2_downsample(fuse_out12) # Downsample fused 1&2
#         if isinstance(self.fuse_mode123, EFC) and out3.shape == fuse_out12_ds.shape:
#             fuse_out123 = self.fuse_mode123(out3, fuse_out12_ds)
#         else:
#             if not isinstance(self.fuse_mode123, EFC): fuse_out123 = out3
#             else:
#                  print(f"Warning: Shape mismatch for fuse_mode123: out3 {out3.shape}, fuse_out12_ds {fuse_out12_ds.shape}. Using out3.")
#                  fuse_out123 = out3


#         # Fuse 1&2&3 result with 4
#         fuse_out123_ds = self.layer3_downsample(fuse_out123) # Downsample fused 1&2&3
#         if isinstance(self.fuse_mode1234, EFC) and out4.shape == fuse_out123_ds.shape:
#             fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_ds)
#         else:
#             if not isinstance(self.fuse_mode1234, EFC): fuse_out1234 = out4
#             else:
#                  print(f"Warning: Shape mismatch for fuse_mode1234: out4 {out4.shape}, fuse_out123_ds {fuse_out123_ds.shape}. Using out4.")
#                  fuse_out1234 = out4


#         # 4. Final Attention (GAM)
#         if isinstance(self.gam, GAM_Attention):
#             att_out = self.gam(fuse_out1234)
#         else: # If GAM is Identity
#             att_out = fuse_out1234

#         # 5. Pooling
#         # Check for zero spatial dimensions before pooling
#         if att_out.numel() == 0 or att_out.shape[2] == 0 or att_out.shape[3] == 0:
#             print(f"Warning: Zero dimension detected before pooling: {att_out.shape}. Returning zero embedding.")
#             return torch.zeros(B, self.embd_dim, device=x.device, dtype=x.dtype)

#         stats = self.pooling(att_out) # Apply selected pooling

#         # 6. Embedding Layers
#         # Check dimension match before linear layer
#         if stats.shape[1] != self.stats_dim_actual:
#              raise ValueError(f"Dimension mismatch before final Linear layer seg_1: "
#                               f"Expected {self.stats_dim_actual}, got {stats.shape[1]}. "
#                               f"Input size: {F}, Final Feature Channels: {att_out.shape[1]}, "
#                               f"Pooling Type: {self.pooling_type}")

#         embed_a = self.seg_1(stats)
#         if self.two_emb_layer:
#             # Apply ReLU, BN, Linear for second embedding layer
#             out_emb = self.relu_emb(embed_a)
#             out_emb = self.seg_bn_1(out_emb)
#             embed_b = self.seg_2(out_emb)
#             return embed_b
#         else:
#             # Return directly after first linear layer
#             return embed_a

# # --- Main Execution Block ---
# if __name__ == "__main__":
#     # 1. Define model and input hyperparameters
#     input_feat_dim = 80     # Frequency dimension (e.g., MFCCs, Fbank)
#     time_steps = 200        # Time dimension (sequence length) - Adjust as needed
#     batch_size = 4          # Example batch size
#     embedding_dim = 192     # Desired output embedding size
#     m_channels_val = 32     # Base channels for the model
#     use_two_emb = False     # Use single or double embedding layer
#     pooling = 'stats'       # Pooling type: 'stats', 'avg', 'attentive'

#     # 2. Instantiate the modified model
#     print("正在实例化 ERes2Net_DSConv 模型 (使用深度可分离卷积)...")
#     try:
#         model = ERes2Net_DSConv(input_size=input_feat_dim,
#                                 embd_dim=embedding_dim,
#                                 m_channels=m_channels_val,
#                                 two_emb_layer=use_two_emb,
#                                 pooling_type=pooling,
#                                 # Can override other defaults like num_blocks, expansion, etc. here
#                                 # num_blocks=[3, 4, 6, 3], # Example
#                                 # expansion=2,           # Example
#                                 # scale=4,               # Example Res2Net scale
#                                )
#         model.eval() # Set model to evaluation mode for profiling/summary
#         print("模型实例化成功！")
#     except Exception as e:
#         print(f"模型实例化失败: {e}")
#         exit()

#     # 3. Create a dummy input tensor
#     # Input shape: (Batch, Time, Freq) for 3D input
#     # Or (Batch, 1, Freq, Time) for 4D input (the model handles both via forward)
#     dummy_input_3d = torch.randn(batch_size, time_steps, input_feat_dim)
#     # dummy_input_4d = torch.randn(batch_size, 1, input_feat_dim, time_steps) # Alternative input format
#     print(f"创建虚拟输入张量，形状: {dummy_input_3d.shape}")

#     # 4. Test forward pass
#     print("测试前向传播...")
#     try:
#         with torch.no_grad(): # No need to track gradients
#             output = model(dummy_input_3d)
#         print(f"前向传播成功！输出形状: {output.shape}")
#         assert output.shape == (batch_size, embedding_dim)
#     except Exception as e:
#         print(f"前向传播失败: {e}")
#         # exit() # Optional: exit if forward pass fails

#     # 5. Use torchsummary for detailed layer info (if available)
#     print("-" * 80)
#     if _torchsummary_available:
#         print("使用 torchsummary 生成模型摘要:")
#         try:
#             # torchsummary expects input_size = (channels, freq_dim, time_steps)
#             # Our model's forward handles permutation, so provide (1, F, T)
#             input_shape_for_summary = (1, input_feat_dim, time_steps)
#             print(f"提供给 torchsummary 的 input_size: {input_shape_for_summary}")
#             torchsummary_summary(model, input_size=input_shape_for_summary, device="cpu")
#         except Exception as e:
#             print(f"\n运行 torchsummary summary 时出错: {e}")
#             # Fallback to basic count if torchsummary fails after being found
#             _torchsummary_available = False # Disable flag to trigger basic count below

#     # 6. Calculate FLOPs and Parameters using thop (if available)
#     # If torchsummary failed or wasn't available, print basic count first
#     if not _torchsummary_available:
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print("torchsummary 不可用或运行失败，显示基本参数计数:")
#         print(f"  Total params: {total_params:,}")
#         print(f"  Trainable params: {trainable_params:,}")
#         print(f"  Non-trainable params: {total_params - trainable_params:,}")

#     print("-" * 80)
#     if _thop_available:
#         print("使用 thop 计算 FLOPs 和参数:")
#         try:
#             # thop needs input as a tuple for profile
#             # Use the 4D input format (B, C, F, T) which the model expects internally
#             thop_input = torch.randn(1, 1, input_feat_dim, time_steps) # Use batch size 1 for standard FLOPs count
#             print(f"提供给 thop 的 input shape: {thop_input.shape}")

#             # Profile the model
#             macs, params = profile(model, inputs=(thop_input,), verbose=False)

#             # Convert MACs to GFLOPs (1 MAC = 2 FLOPs approx.)
#             gflops = (macs * 2) / 1e9

#             print(f"  计算得到的参数量 (来自 thop): {params:,}")
#             print(f"  估计的 Multiply-Accumulate Operations (MACs): {macs:,}")
#             print(f"  估计的 Floating Point Operations (GFLOPs): {gflops:.2f} GFLOPs")

#             # Verify thop params vs manual count (should be close)
#             manual_total_params = sum(p.numel() for p in model.parameters())
#             if params != manual_total_params:
#                  print(f"  注意: thop 参数计数 ({params:,}) 与手动计数 ({manual_total_params:,}) 不同。")

#         except Exception as e:
#             print(f"\n运行 thop profile 时出错: {e}")
#             print("  无法计算 FLOPs。")
#     else:
#         print("thop 库未安装，跳过 FLOPs 计算。")

#     print("-" * 80)
