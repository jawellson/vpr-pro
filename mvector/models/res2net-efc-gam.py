import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as torchsummary_summary
from mvector.models.pooling import TemporalAveragePooling, TemporalStatsPool, AttentiveStatisticsPooling
from thop import profile
# class TemporalStatsPool(nn.Module):

#     def __init__(self):
#         super(TemporalStatsPool, self).__init__()

#     def forward(self, x):
#         # 确保张量有空间维度
#         if x.dim() < 4 or x.shape[2] == 0 or x.shape[3] == 0:
#              print(f"警告: TemporalStatsPool 收到形状为 {x.shape} 的输入。返回零张量。")
#              b, c = x.shape[0], x.shape[1]
#              return torch.zeros(b, c * 2, device=x.device)

#         # 获取批次大小和通道数 - 这一行缺失导致了错误
#         b, c, h, w = x.shape
        
#         # 假设标准语谱图(B, C, 频率, 时间) -> 在时间维度上池化(维度3)
#         epsilon = 1e-10
#         mean = x.mean(dim=3) # 时间维度上的均值
#         std = x.std(dim=3).clamp(min=epsilon) # 时间维度上的标准差
#         # 合并频率统计：在频率维度上展平均值/标准差
#         stats = torch.cat((mean.view(b, -1), std.view(b, -1)), dim=1)
#         return stats


# --- 深度可分离卷积模块 ---
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积模块：深度卷积 -> 逐点卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        in_channels, out_channels = max(1, in_channels), max(1, out_channels)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# --- 修改后的卷积定义 ---
def conv1x1(in_planes, out_planes, stride=1):
    """无填充的1x1卷积（逐点）"""
    in_planes, out_planes = max(1, in_planes), max(1, out_planes)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """带填充的3x3深度可分离卷积"""
    in_planes, out_planes = max(1, in_planes), max(1, out_planes)
    return DepthwiseSeparableConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# --- 其他类定义 ---
class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'

# --- 使用DSConv但保持原始通道路径分割的EFC模块 ---
class EFC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c1, c2 = max(c1, 1), max(c2, 1)
        # 使用合理的默认值，避免c2较小时除零
        num_groups_default = 4
        if c2 > 0 and c2 // num_groups_default > 0:
            self.group_num = min(num_groups_default, c2 // (c2 // num_groups_default))
        else:
            self.group_num = 1

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.bn_conv1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)
        self.bn_conv2 = nn.BatchNorm2d(c2)
        self.conv4_global = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.ones(c2, 1, 1))  # 将gamma初始化为1
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))  # 将beta初始化为0

        # 门控生成器 - 保持修改版的实现
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, max(c2 // 4, 1), 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(max(c2 // 4, 1), c2, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 深度卷积部分
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False)
        self.bn_dwconv = nn.BatchNorm2d(c2)
        self.relu_dwconv = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)
        self.bn_conv3 = nn.BatchNorm2d(c2)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)

        # 使用深度可分离卷积的交互部分
        interact_channels = max(c2 // self.group_num, 1) if self.group_num > 0 else c2
        if self.group_num > 0 and interact_channels > 0:
            self.interacts = nn.ModuleList([
                DepthwiseSeparableConv(interact_channels, interact_channels, kernel_size=3, 
                                       stride=1, padding=1, bias=False)
                for _ in range(self.group_num)
            ])
            self.bn_interact = nn.ModuleList([
                nn.BatchNorm2d(interact_channels) for _ in range(self.group_num)
            ])
        else:
            self.interacts = nn.ModuleList()
            self.bn_interact = nn.ModuleList()
        self.relu_interact = ReLU(inplace=True)

    def forward(self, x1, x2):
        # 初始卷积和融合
        global_conv1 = self.conv1(x1); bn_x1 = self.bn_conv1(global_conv1)
        global_conv2 = self.conv2(x2); bn_x2 = self.bn_conv2(global_conv2)
        
        # 计算原版的weight_1和weight_2
        weight_1 = self.sigmoid(bn_x1).clamp(min=self.eps, max=1.0 - self.eps)
        weight_2 = self.sigmoid(bn_x2).clamp(min=self.eps, max=1.0 - self.eps)
        
        X_GLOBAL = bn_x1 + bn_x2  # 融合特征
        x_conv4_global = self.conv4_global(X_GLOBAL)

        # 分组交互与DSConv
        if self.group_num > 0 and len(self.interacts) > 0 and X_GLOBAL.size(1) > 0 and X_GLOBAL.size(1) % self.group_num == 0:
            interact_channels = X_GLOBAL.size(1) // self.group_num
            X_chunks = X_GLOBAL.chunk(self.group_num, dim=1)
            out_interact = []
            for group_id in range(self.group_num):
                chunk = X_chunks[group_id]
                out_chunk = self.interacts[group_id](chunk)
                out_chunk = self.bn_interact[group_id](out_chunk)
                out_chunk = self.relu_interact(out_chunk)
                out_interact.append(out_chunk)
            out_interact_cat = torch.cat(out_interact, dim=1)
        else:
            out_interact_cat = X_GLOBAL

        # 组归一化部分
        N, C, H, W = out_interact_cat.size()
        if self.group_num > 0 and C > 0 and C % self.group_num == 0:
            x_group = out_interact_cat.view(N, self.group_num, C // self.group_num, H, W)
            mean = x_group.mean(dim=(2, 3, 4), keepdim=True)
            std = x_group.std(dim=(2, 3, 4), keepdim=True).clamp(min=self.eps)
            x_norm = (x_group - mean) / std
            x_norm = x_norm.view(N, C, H, W)
            if self.gamma.shape[0] == C:
                x_gui = x_norm * self.gamma + self.beta
            else:
                x_gui = x_norm
        else:
            x_gui = out_interact_cat

        # 恢复原版的通道路径分割方式
        weight_x3 = self.Apt(X_GLOBAL)
        reweights = self.sigmoid(weight_x3).clamp(min=self.eps, max=1.0 - self.eps)
        
        # 原版的通道路径分割逻辑
        x_up_1 = (reweights >= weight_1).float()
        x_low_1 = (reweights < weight_1).float()
        x_up_2 = (reweights >= weight_2).float()
        x_low_2 = (reweights < weight_2).float()
        x_up = x_up_1 * X_GLOBAL + x_up_2 * X_GLOBAL
        x_low = x_low_1 * X_GLOBAL + x_low_2 * X_GLOBAL

        # 下路径处理（深度卷积）
        x_low_dw = self.dwconv(x_low)
        x_low_dw = self.bn_dwconv(x_low_dw)
        x_low_dw = self.relu_dwconv(x_low_dw)
        x_low_pw = self.conv3(x_low_dw)
        x_low_pw = self.bn_conv3(x_low_pw)
        gate = self.gate_generator(x_low)
        x_low_gated = x_low_pw * gate

        # 上路径处理（逐点卷积）
        x_up_pw = self.conv4(x_up)

        # 合并路径并添加归一化特征
        xL = x_low_gated + x_up_pw + x_gui
        xL = torch.nan_to_num(xL, nan=0.0, posinf=1e5, neginf=-1e5)
        return xL


# --- 使用DSConv的修改版GAM注意力 ---
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()
        in_channels = max(in_channels, 1)
        inter_channels = max(in_channels // rate, 1)

        # 通道注意力(MLP)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, in_channels)
        )

        # 空间注意力 - 使用深度可分离卷积
        self.spatial_attention = nn.Sequential(
            DepthwiseSeparableConv(in_channels, inter_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(inter_channels, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 0:
            return x

        # 通道注意力路径
        channel_descriptor = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)

        if torch.abs(channel_descriptor).sum() < 1e-9:
            channel_att_weight = torch.ones(b, c, 1, 1, device=x.device)
        else:
            channel_att_raw = self.channel_attention(channel_descriptor).view(b, c, 1, 1)
            channel_att_weight = self.sigmoid(channel_att_raw)

        x_channel_att = x * channel_att_weight

        # 空间注意力路径
        spatial_att_raw = self.spatial_attention(x_channel_att)
        spatial_att_weight = self.sigmoid(spatial_att_raw)

        out = x_channel_att * spatial_att_weight
        return out

# --- 使用DSConv的重命名Res2Net块 ---
class Res2NetBlock(nn.Module):
    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(Res2NetBlock, self).__init__()
        self.expansion = expansion
        width = max(int(math.floor(planes * (base_width / 64.0))), 1)
        out_c_conv1 = max(width * scale, 1)
        in_planes = max(in_planes, 1)

        self.conv1 = conv1x1(in_planes, out_c_conv1, stride)
        self.bn1 = nn.BatchNorm2d(out_c_conv1)
        self.relu = ReLU(inplace=True)
        self.nums = max(scale, 1)
        self.width = width

        convs = []; bns = []
        for _ in range(self.nums):
            convs.append(conv3x3(self.width, self.width))
            bns.append(nn.BatchNorm2d(self.width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        out_c_conv3 = max(planes * self.expansion, 1)
        in_c_conv3 = max(self.width * self.nums, 1)

        self.conv3 = conv1x1(in_c_conv3, out_c_conv3)
        self.bn3 = nn.BatchNorm2d(out_c_conv3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_c_conv3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_c_conv3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c_conv3)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

        if self.nums <= 0 or self.width <= 0 or out.size(1) < self.width * self.nums:
            if self.nums > 0 and len(self.convs) > 0:
                sp = self.convs[0](out); sp = self.bns[0](sp); processed_out = self.relu(sp)
            else:
                processed_out = out
            y = [processed_out]
        else:
            spx = torch.split(out, self.width, 1)
            y = []
            for i in range(self.nums):
                if i == 0:
                    sp = spx[i]
                else:
                    if y and y[-1].size() == spx[i].size():
                        sp = y[-1] + spx[i]
                    else:
                        sp = spx[i]

                sp = self.convs[i](sp); sp = self.bns[i](sp); sp = self.relu(sp)
                y.append(sp)

        if not y:
            out = torch.tensor([], device=x.device, dtype=x.dtype)
        else:
            out = torch.cat(y, 1)

        if out.numel() > 0 and out.size(1) == self.conv3.in_channels:
            out = self.conv3(out)
            out = self.bn3(out)
            if out.shape == shortcut.shape:
                out = out + shortcut
            else:
                print(f"警告: Res2NetBlock输出形状 {out.shape} != shortcut形状 {shortcut.shape}。跳过残差连接。")
        else:
            out = shortcut

        out = self.relu(out)
        return out

# --- 使用DSConv和修改的EFC的重命名EFC融合块 ---
class EFCFusionRes2NetBlock(nn.Module):
    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(EFCFusionRes2NetBlock, self).__init__()
        self.expansion = expansion
        width = max(int(math.floor(planes * (base_width / 64.0))), 1)
        out_c_conv1 = max(width * scale, 1)
        in_planes = max(in_planes, 1)
        planes = max(planes, 1)

        self.conv1 = conv1x1(in_planes, out_c_conv1, stride)
        self.bn1 = nn.BatchNorm2d(out_c_conv1)
        self.relu = ReLU(inplace=True)
        self.nums = max(scale, 1)
        self.width = width

        convs = []; bns = []
        for i in range(self.nums):
            convs.append(conv3x3(self.width, self.width))
            bns.append(nn.BatchNorm2d(self.width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        fuse_models = []
        for _ in range(self.nums - 1):
            if self.width > 0:
                fuse_models.append(EFC(c1=self.width, c2=self.width))
            else:
                fuse_models.append(nn.Identity())
        self.fuse_models = nn.ModuleList(fuse_models)

        out_c_conv3 = max(planes * self.expansion, 1)
        in_c_conv3 = max(self.width * self.nums, 1)

        self.conv3 = conv1x1(in_c_conv3, out_c_conv3)
        self.bn3 = nn.BatchNorm2d(out_c_conv3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_c_conv3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_c_conv3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c_conv3)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

        if self.nums <= 0 or self.width <= 0 or out.size(1) < self.width * self.nums:
            if self.nums > 0 and len(self.convs) > 0:
                sp = self.convs[0](out); sp = self.bns[0](sp); processed_out = self.relu(sp)
            else:
                processed_out = out
            y = [processed_out]
        else:
            spx = torch.split(out, self.width, 1)
            y = []
            last_processed = None
            for i in range(self.nums):
                current_input = spx[i]
                if i > 0 and i-1 < len(self.fuse_models) and last_processed is not None:
                    if isinstance(self.fuse_models[i - 1], EFC) and \
                       last_processed.size(1) == self.width and \
                       current_input.size(1) == self.width:
                        sp = self.fuse_models[i - 1](last_processed, current_input)
                    else:
                        sp = current_input
                else:
                    sp = current_input

                sp = self.convs[i](sp); sp = self.bns[i](sp); sp = self.relu(sp)
                y.append(sp)
                last_processed = sp

        if not y:
            out = torch.tensor([], device=x.device, dtype=x.dtype)
        else:
            out = torch.cat(y, 1)

        if out.numel() > 0 and out.size(1) == self.conv3.in_channels:
            out = self.conv3(out)
            out = self.bn3(out)
            if out.shape == shortcut.shape:
                out = out + shortcut
            else:
                print(f"警告: EFCFusionRes2NetBlock输出形状 {out.shape} != shortcut形状 {shortcut.shape}。跳过残差连接。")
        else:
            out = shortcut

        out = self.relu(out)
        return out

# --- 主模型: ERes2Net_DSConv (修改后的ERes2Net) ---
class ERes2Net_DSConv(nn.Module):
    def __init__(self,
                 input_size,
                 block=Res2NetBlock,
                 block_fuse=EFCFusionRes2NetBlock,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 expansion=2,
                 base_width=32,
                 scale=2,
                 embd_dim=192,
                 two_emb_layer=False,
                 pooling_type='stats'):
        super(ERes2Net_DSConv, self).__init__()

        input_size = max(input_size, 1)
        m_channels = max(m_channels, 1)
        embd_dim = max(embd_dim, 1)

        self.in_planes = m_channels
        self.expansion = expansion
        self.embd_dim = embd_dim
        self.two_emb_layer = two_emb_layer

        # 初始卷积：使用深度可分离卷积
        self.conv1 = DepthwiseSeparableConv(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.relu1 = ReLU(inplace=True)

        # --- 骨干网络层 ---
        layer_out_channels = []
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)

        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)

        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)

        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)

        l1_out_c, l2_out_c, l3_out_c, l4_out_c = layer_out_channels

        # --- 下采样和融合层 ---
        # 下采样第1层输出以匹配第2层通道大小
        ds1_out_c = max(l2_out_c, 1)
        self.layer1_downsample = nn.Sequential(
            DepthwiseSeparableConv(max(l1_out_c, 1), ds1_out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ds1_out_c), ReLU(inplace=True)
        ) if l1_out_c > 0 else nn.Identity()
        self.fuse_mode12 = EFC(c1=ds1_out_c, c2=ds1_out_c) if ds1_out_c > 0 else nn.Identity()

        # 下采样融合的第1&2层输出以匹配第3层通道大小
        ds2_out_c = max(l3_out_c, 1)
        self.layer2_downsample = nn.Sequential(
            DepthwiseSeparableConv(ds1_out_c, ds2_out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ds2_out_c), ReLU(inplace=True)
        ) if ds1_out_c > 0 else nn.Identity()
        self.fuse_mode123 = EFC(c1=ds2_out_c, c2=ds2_out_c) if ds2_out_c > 0 else nn.Identity()

        # 下采样融合的第1&2&3层输出以匹配第4层通道大小
        ds3_out_c = max(l4_out_c, 1)
        self.layer3_downsample = nn.Sequential(
            DepthwiseSeparableConv(ds2_out_c, ds3_out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ds3_out_c), ReLU(inplace=True)
        ) if ds2_out_c > 0 else nn.Identity()
        self.fuse_mode1234 = EFC(c1=ds3_out_c, c2=ds3_out_c) if ds3_out_c > 0 else nn.Identity()

        # 最终注意力模块
        self.gam = GAM_Attention(in_channels=ds3_out_c) if ds3_out_c > 0 else nn.Identity()
        final_feature_channels = ds3_out_c

        # --- 池化层 ---
        self.pooling_type = pooling_type
        if pooling_type == 'stats':
            self.pooling = TemporalStatsPool()
            freq_dim_after_conv = math.ceil(input_size / 8.0)
            self.stats_dim_actual = max(int(2 * final_feature_channels * freq_dim_after_conv), 1)
        elif pooling_type == 'avg':
            self.pooling = TemporalAveragePooling()
            self.stats_dim_actual = max(final_feature_channels, 1)
        elif pooling_type == 'attentive':
            self.pooling = AttentiveStatisticsPooling(final_feature_channels, 128)
            self.stats_dim_actual = max(2 * final_feature_channels, 1)
        else:
            raise ValueError(f"不支持的池化类型: {pooling_type}")

        # --- 嵌入层 ---
        self.seg_1 = nn.Linear(self.stats_dim_actual, embd_dim)
        if self.two_emb_layer:
            self.relu_emb = nn.ReLU(inplace=True)
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.relu_emb = nn.Identity()
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                except ValueError:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_planes = max(planes, 1)

        try:
            layers.append(block(self.expansion, self.in_planes, current_planes, stride=strides[0], base_width=base_width, scale=scale))
            self.in_planes = current_planes * self.expansion
        except Exception as e:
            print(f"创建层中第一个块时出错: {e}")
            layers.append(nn.Identity())

        for i in range(1, num_blocks):
            try:
                if self.in_planes > 0:
                    layers.append(block(self.expansion, self.in_planes, current_planes, stride=strides[i], base_width=base_width, scale=scale))
                else:
                    layers.append(nn.Identity())
            except Exception as e:
                print(f"创建层中第{i}个块时出错: {e}")
                layers.append(nn.Identity())

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状检查
        if x.dim() == 3:
            # 假设 (B, T, F) -> 转换为 (B, F, T)
            x = x.permute(0, 2, 1)
        elif x.dim() == 4:
            # 假设 (B, 1, F, T) - 保持原样
            pass
        else:
            raise ValueError(f"期望3D (B, T, F) 或 4D (B, 1, F, T) 输入, 得到 {x.dim()}D 形状 {x.shape}")

        B, C, F, T = x.shape
        if F == 0 or T == 0:
            print(f"警告: 输入中检测到零维度 F={F}, T={T}。返回零嵌入。")
            return torch.zeros(B, self.embd_dim, device=x.device, dtype=x.dtype)

        # 确保输入有1个通道
        if C != 1:
            x = x.unsqueeze(1)

        # --- 前向传播 ---
        # 1. 初始DSConv + BN + ReLU
        out = self.conv1(x); out = self.bn1(out); out = self.relu1(out)

        # 2. 骨干网络层
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # 3. 融合路径
        # 融合1和2
        out1_ds = self.layer1_downsample(out1)
        if isinstance(self.fuse_mode12, EFC) and out2.shape == out1_ds.shape:
            fuse_out12 = self.fuse_mode12(out2, out1_ds)
        else:
            if not isinstance(self.fuse_mode12, EFC): 
                fuse_out12 = out2
            else:
                print(f"警告: fuse_mode12形状不匹配: out2 {out2.shape}, out1_ds {out1_ds.shape}。使用out2。")
                fuse_out12 = out2

        # 融合1&2结果与3
        fuse_out12_ds = self.layer2_downsample(fuse_out12)
        if isinstance(self.fuse_mode123, EFC) and out3.shape == fuse_out12_ds.shape:
            fuse_out123 = self.fuse_mode123(out3, fuse_out12_ds)
        else:
            if not isinstance(self.fuse_mode123, EFC): 
                fuse_out123 = out3
            else:
                print(f"警告: fuse_mode123形状不匹配: out3 {out3.shape}, fuse_out12_ds {fuse_out12_ds.shape}。使用out3。")
                fuse_out123 = out3

        # 融合1&2&3结果与4
        fuse_out123_ds = self.layer3_downsample(fuse_out123)
        if isinstance(self.fuse_mode1234, EFC) and out4.shape == fuse_out123_ds.shape:
            fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_ds)
        else:
            if not isinstance(self.fuse_mode1234, EFC): 
                fuse_out1234 = out4
            else:
                print(f"警告: fuse_mode1234形状不匹配: out4 {out4.shape}, fuse_out123_ds {fuse_out123_ds.shape}。使用out4。")
                fuse_out1234 = out4

        # 4. 最终注意力
        if isinstance(self.gam, GAM_Attention):
            att_out = self.gam(fuse_out1234)
        else:
            att_out = fuse_out1234

        # 5. 池化
        if att_out.numel() == 0 or att_out.shape[2] == 0 or att_out.shape[3] == 0:
            print(f"警告: 池化前检测到零维度: {att_out.shape}。返回零嵌入。")
            return torch.zeros(B, self.embd_dim, device=x.device, dtype=x.dtype)

        stats = self.pooling(att_out)

        # 6. 嵌入层
        if stats.shape[1] != self.stats_dim_actual:
            raise ValueError(f"seg_1线性层前维度不匹配: "
                           f"期望 {self.stats_dim_actual}, 得到 {stats.shape[1]}. "
                           f"输入大小: {F}, 最终特征通道: {att_out.shape[1]}, "
                           f"池化类型: {self.pooling_type}")

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            # 应用ReLU, BN, Linear为第二个嵌入层
            out_emb = self.relu_emb(embed_a)
            out_emb = self.seg_bn_1(out_emb)
            embed_b = self.seg_2(out_emb)
            return embed_b
        else:
            # 直接在第一个线性层后返回
            return embed_a

# --- 主执行块 ---
if __name__ == "__main__":
    # 1. 定义模型和输入超参数
    input_feat_dim = 80     # 频率维度 (例如, MFCCs, Fbank)
    time_steps = 200        # 时间维度 (序列长度) - 根据需要调整
    batch_size = 4          # 示例批大小
    embedding_dim = 192     # 期望的输出嵌入大小
    m_channels_val = 32     # 模型的基础通道数
    use_two_emb = False     # 使用单一或双嵌入层
    pooling = 'stats'       # 池化类型: 'stats', 'avg', 'attentive'

    # 2. 实例化修改后的模型
    print("正在实例化 ERes2Net_DSConv 模型 (使用深度可分离卷积)...")
    try:
        model = ERes2Net_DSConv(input_size=input_feat_dim,
                                embd_dim=embedding_dim,
                                m_channels=m_channels_val,
                                two_emb_layer=use_two_emb,
                                pooling_type=pooling,
                                # 可在此处覆盖其他默认值如num_blocks, expansion等
                                # num_blocks=[3, 4, 6, 3], # 示例
                                # expansion=2,           # 示例
                                # scale=4,               # 示例Res2Net尺度
                               )
        model.eval() # 设置模型为评估模式以进行分析/总结
        print("模型实例化成功！")
    except Exception as e:
        print(f"模型实例化失败: {e}")
        exit()

    # 3. 创建一个模拟输入张量
    # 输入形状: (批次, 时间, 频率) 对于3D输入
    # 或 (批次, 1, 频率, 时间) 对于4D输入(通过forward方法处理两种形式)
    dummy_input_3d = torch.randn(batch_size, time_steps, input_feat_dim)
    # dummy_input_4d = torch.randn(batch_size, 1, input_feat_dim, time_steps) # 替代输入格式
    print(f"创建虚拟输入张量，形状: {dummy_input_3d.shape}")

    # 4. 测试前向传播
    print("测试前向传播...")
    try:
        with torch.no_grad(): # 无需跟踪梯度
            output = model(dummy_input_3d)
        print(f"前向传播成功！输出形状: {output.shape}")
        assert output.shape == (batch_size, embedding_dim)
    except Exception as e:
        print(f"前向传播失败: {e}")
        # exit() # 可选: 如果前向传播失败则退出

    # 5. 使用torchsummary获取详细层信息(如果可用)
    print("-" * 80)
    if _torchsummary_available:
        print("使用 torchsummary 生成模型摘要:")
        try:
            # torchsummary期望input_size = (通道, 频率维度, 时间步长)
            # 我们的模型的forward处理置换，所以提供(1, F, T)
            input_shape_for_summary = (1, input_feat_dim, time_steps)
            print(f"提供给 torchsummary 的 input_size: {input_shape_for_summary}")
            torchsummary_summary(model, input_size=input_shape_for_summary, device="cpu")
        except Exception as e:
            print(f"\n运行 torchsummary summary 时出错: {e}")
            # 如果torchsummary在被发现后失败，回退到基本计数
            _torchsummary_available = False # 禁用标志以触发下面的基本计数

    # 6. 使用thop计算FLOPs和参数(如果可用)
    # 如果torchsummary失败或不可用，先打印基本计数
    if not _torchsummary_available:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("torchsummary 不可用或运行失败，显示基本参数计数:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  非可训练参数: {total_params - trainable_params:,}")

    print("-" * 80)
    if _thop_available:
        print("使用 thop 计算 FLOPs 和参数:")
        try:
            # thop需要输入作为profile的元组
            # 使用模型内部期望的4D输入格式(B, C, F, T)
            thop_input = torch.randn(1, 1, input_feat_dim, time_steps) # 使用批大小1进行标准FLOPs计数
            print(f"提供给 thop 的 input shape: {thop_input.shape}")

            # 分析模型
            macs, params = profile(model, inputs=(thop_input,), verbose=False)

            # 将MACs转换为GFLOPs(1 MAC ≈ 2 FLOPs)
            gflops = (macs * 2) / 1e9

            print(f"  计算得到的参数量 (来自 thop): {params:,}")
            print(f"  估计的 Multiply-Accumulate Operations (MACs): {macs:,}")
            print(f"  估计的 Floating Point Operations (GFLOPs): {gflops:.2f} GFLOPs")

            # 验证thop参数与手动计数(应该接近)
            manual_total_params = sum(p.numel() for p in model.parameters())
            if params != manual_total_params:
                print(f"  注意: thop 参数计数 ({params:,}) 与手动计数 ({manual_total_params:,}) 不同。")

        except Exception as e:
            print(f"\n运行 thop profile 时出错: {e}")
            print("  无法计算 FLOPs。")
    else:
        print("thop 库未安装，跳过 FLOPs 计算。")

    print("-" * 80)
