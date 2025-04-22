# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 torchsummary
try:
    # Alias to avoid potential name conflict if user also defines 'summary'
    from torchsummary import summary as torchsummary_summary
    _torchsummary_available = True
    print("torchsummary 库已找到。")
except ImportError:
    print("警告：未找到 torchsummary 库。将无法打印详细模型信息。")
    print("请运行 'pip install torchsummary' 来安装。")
    _torchsummary_available = False

# --- 深度可分离卷积模块 ---
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块：Depthwise Conv -> Pointwise Conv
    不包含内部的BN和ReLU，以便灵活替换标准卷积层。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        in_channels = max(1, in_channels)
        out_channels = max(1, out_channels)
        # 深度卷积: 每个输入通道一个滤波器
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        # 逐点卷积: 1x1 卷积混合通道信息
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# --- 修改后的卷积定义 ---
def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution (Pointwise) without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 depthwise separable convolution with padding"
    # 使用深度可分离卷积
    return DepthwiseSeparableConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# --- 其他类定义 ---
class TemporalStatsPool(nn.Module):
    """ Aggregates statistics over the time dimension """
    def __init__(self):
        super(TemporalStatsPool, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        if h * w == 0:
             return torch.zeros(b, c * 2, device=x.device)
        x_reshaped = x.view(b, c * h, w)
        epsilon = 1e-10
        mean = x_reshaped.mean(dim=2)
        std = x_reshaped.std(dim=2).clamp(min=epsilon)
        stats = torch.cat((mean, std), dim=1)
        return stats

class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)
    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'

# EFC 保持不变
class EFC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c1 = max(c1, 1); c2 = max(c2, 1)
        num_groups_default = 4
        self.group_num = min(num_groups_default, c2 // (c2 // num_groups_default) if c2 > 0 and c2 // num_groups_default > 0 else 1)

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False); self.bn_conv1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False); self.bn_conv2 = nn.BatchNorm2d(c2)
        self.conv4_global = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid(); self.eps = 1e-10
        self.gamma = nn.Parameter(torch.ones(c2, 1, 1)); self.beta = nn.Parameter(torch.zeros(c2, 1, 1))

        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, max(c2 // 4, 1), 1, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(max(c2 // 4, 1), c2, 1, 1, bias=False), nn.Sigmoid())

        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False)
        self.bn_dwconv = nn.BatchNorm2d(c2); self.relu_dwconv = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False); self.bn_conv3 = nn.BatchNorm2d(c2)
        self.Apt = nn.AdaptiveAvgPool2d(1); self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)

        interact_channels = max(c2 // self.group_num, 1) if self.group_num > 0 else c2
        # Interaction uses DepthwiseSeparableConv
        self.interacts = nn.ModuleList([
            DepthwiseSeparableConv(interact_channels, interact_channels, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(self.group_num)
        ]) if self.group_num > 0 and interact_channels > 0 else nn.ModuleList()
        self.bn_interact = nn.ModuleList([
             nn.BatchNorm2d(interact_channels) for _ in range(self.group_num)
        ]) if self.group_num > 0 and interact_channels > 0 else nn.ModuleList()
        self.relu_interact = ReLU(inplace=True)

    def forward(self, x1, x2):
        global_conv1 = self.conv1(x1); bn_x1 = self.bn_conv1(global_conv1)
        global_conv2 = self.conv2(x2); bn_x2 = self.bn_conv2(global_conv2)
        X_GLOBAL = bn_x1 + bn_x2
        x_conv4_global = self.conv4_global(X_GLOBAL)

        if self.group_num > 0 and len(self.interacts) > 0 and X_GLOBAL.size(1) == X_GLOBAL.size(1) // self.group_num * self.group_num:
            interact_channels = X_GLOBAL.size(1) // self.group_num
            X_chunks = X_GLOBAL.chunk(self.group_num, dim=1); out_interact = []
            for group_id in range(self.group_num):
                chunk = X_chunks[group_id]; out_chunk = self.interacts[group_id](chunk)
                out_chunk = self.bn_interact[group_id](out_chunk); out_chunk = self.relu_interact(out_chunk)
                out_interact.append(out_chunk)
            out_interact_cat = torch.cat(out_interact, dim=1)
        else: out_interact_cat = X_GLOBAL

        N, C, H, W = out_interact_cat.size()
        if self.group_num > 0 and C > 0 and C % self.group_num == 0:
             x_group = out_interact_cat.view(N, self.group_num, C // self.group_num, H, W)
             mean = x_group.mean(dim=(2, 3, 4), keepdim=True); std = x_group.std(dim=(2, 3, 4), keepdim=True).clamp(min=self.eps)
             x_norm = (x_group - mean) / std; x_norm = x_norm.view(N, C, H, W)
             if self.gamma.shape[0] == C: x_gui = x_norm * self.gamma + self.beta
             else: x_gui = x_norm
        else: x_gui = out_interact_cat

        reweights = self.sigmoid(self.Apt(X_GLOBAL))
        x_up = X_GLOBAL * reweights; x_low = X_GLOBAL * (1 - reweights)

        x_low_dw = self.dwconv(x_low); x_low_dw = self.bn_dwconv(x_low_dw); x_low_dw = self.relu_dwconv(x_low_dw)
        x_low_pw = self.conv3(x_low_dw); x_low_pw = self.bn_conv3(x_low_pw)
        gate = self.gate_generator(x_low); x_low_gated = x_low_pw * gate
        x_up_pw = self.conv4(x_up)

        xL = x_low_gated + x_up_pw + x_gui
        xL = torch.nan_to_num(xL, nan=0.0, posinf=1e5, neginf=-1e5)
        return xL

# GAM_Attention uses DSConv
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()
        in_channels = max(in_channels, 1)
        inter_channels = max(in_channels // rate, 1)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, inter_channels), nn.ReLU(inplace=True),
            nn.Linear(inter_channels, in_channels))

        self.spatial_attention = nn.Sequential(
            DepthwiseSeparableConv(in_channels, inter_channels, kernel_size=7, padding=3, bias=False), # DSConv
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            DepthwiseSeparableConv(inter_channels, in_channels, kernel_size=7, padding=3, bias=False), # DSConv
            nn.BatchNorm2d(in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape;
        if c == 0: return x
        channel_descriptor = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        if channel_descriptor.abs().sum() < 1e-9: channel_att_weight = torch.ones(b, c, 1, 1, device=x.device)
        else:
             channel_att_raw = self.channel_attention(channel_descriptor).view(b, c, 1, 1)
             channel_att_weight = self.sigmoid(channel_att_raw)
        x_channel_att = x * channel_att_weight
        spatial_att_raw = self.spatial_attention(x_channel_att)
        spatial_att_weight = self.sigmoid(spatial_att_raw)
        out = x_channel_att * spatial_att_weight
        return out

# Renamed: BasicBlockERes2Net -> Res2NetBlock
class Res2NetBlock(nn.Module):
    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(Res2NetBlock, self).__init__()
        self.expansion = expansion
        width = max(int(math.floor(planes * (base_width / 64.0))), 1)
        out_c_conv1 = max(width * scale, 1); in_planes = max(in_planes, 1)

        self.conv1 = conv1x1(in_planes, out_c_conv1, stride)
        self.bn1 = nn.BatchNorm2d(out_c_conv1); self.relu = ReLU(inplace=True)
        self.nums = max(scale, 1); self.width = width

        convs = []; bns = []
        for _ in range(self.nums):
            convs.append(conv3x3(self.width, self.width)) # Uses DSConv via conv3x3
            bns.append(nn.BatchNorm2d(self.width))
        self.convs = nn.ModuleList(convs); self.bns = nn.ModuleList(bns)

        out_c_conv3 = max(planes * self.expansion, 1)
        in_c_conv3 = max(self.width * self.nums, 1)
        self.conv3 = conv1x1(in_c_conv3, out_c_conv3)
        self.bn3 = nn.BatchNorm2d(out_c_conv3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_c_conv3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_c_conv3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c_conv3))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

        if self.nums <= 0 or self.width <= 0 or out.size(1) < self.width * self.nums:
            if self.nums > 0 and len(self.convs) > 0:
                 sp = self.convs[0](out); sp = self.bns[0](sp); processed_out = self.relu(sp)
            else: processed_out = out
            y = [processed_out]
        else:
            spx = torch.split(out, self.width, 1); y = []
            for i in range(self.nums):
                if i == 0: sp = spx[i]
                else:
                    if y and y[-1].size() == spx[i].size(): sp = y[-1] + spx[i]
                    else: sp = spx[i]
                sp = self.convs[i](sp); sp = self.bns[i](sp); sp = self.relu(sp)
                y.append(sp)

        if not y: out = torch.tensor([], device=x.device)
        else: out = torch.cat(y, 1)

        if out.numel() > 0 and out.size(1) == self.conv3.in_channels:
             out = self.conv3(out); out = self.bn3(out); out = out + shortcut
        else: out = shortcut
        out = self.relu(out)
        return out

# Renamed: BasicBlockERes2Net_diff_AFF -> EFCFusionRes2NetBlock
class EFCFusionRes2NetBlock(nn.Module):
    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(EFCFusionRes2NetBlock, self).__init__()
        self.expansion = expansion
        width = max(int(math.floor(planes * (base_width / 64.0))), 1)
        out_c_conv1 = max(width * scale, 1); in_planes = max(in_planes, 1); planes = max(planes, 1)

        self.conv1 = conv1x1(in_planes, out_c_conv1, stride)
        self.bn1 = nn.BatchNorm2d(out_c_conv1); self.relu = ReLU(inplace=True)
        self.nums = max(scale, 1); self.width = width

        convs = []; bns = []
        for i in range(self.nums):
            convs.append(conv3x3(self.width, self.width)) # Uses DSConv via conv3x3
            bns.append(nn.BatchNorm2d(self.width))
        self.convs = nn.ModuleList(convs); self.bns = nn.ModuleList(bns)

        fuse_models = []
        for _ in range(self.nums - 1):
            if self.width > 0: fuse_models.append(EFC(c1=self.width, c2=self.width))
            else: fuse_models.append(nn.Identity())
        self.fuse_models = nn.ModuleList(fuse_models)

        out_c_conv3 = max(planes * self.expansion, 1)
        in_c_conv3 = max(self.width * self.nums, 1)
        self.conv3 = conv1x1(in_c_conv3, out_c_conv3)
        self.bn3 = nn.BatchNorm2d(out_c_conv3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_c_conv3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_c_conv3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c_conv3))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)

        if self.nums <= 0 or self.width <= 0 or out.size(1) < self.width * self.nums:
            if self.nums > 0 and len(self.convs) > 0:
                 sp = self.convs[0](out); sp = self.bns[0](sp); processed_out = self.relu(sp)
            else: processed_out = out
            y = [processed_out]
        else:
            spx = torch.split(out, self.width, 1); y = []; last_processed = None
            for i in range(self.nums):
                current_input = spx[i]
                if i > 0 and i-1 < len(self.fuse_models) and last_processed is not None:
                     if isinstance(self.fuse_models[i - 1], EFC) and last_processed.size(1) == self.width and current_input.size(1) == self.width:
                        sp = self.fuse_models[i - 1](last_processed, current_input)
                     else: sp = current_input
                else: sp = current_input
                sp = self.convs[i](sp); sp = self.bns[i](sp); sp = self.relu(sp)
                y.append(sp); last_processed = sp

        if not y: out = torch.tensor([], device=x.device)
        else: out = torch.cat(y, 1)

        if out.numel() > 0 and out.size(1) == self.conv3.in_channels:
            out = self.conv3(out); out = self.bn3(out); out = out + shortcut
        else: out = shortcut
        out = self.relu(out)
        return out

# Renamed: ERes2Net -> EFCRes2Net
class EFCRes2Net(nn.Module):
    def __init__(self,
                 input_size,
                 # Updated default block names
                 block=Res2NetBlock,
                 block_fuse=EFCFusionRes2NetBlock,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 expansion=2,
                 base_width=32,
                 scale=2,
                 embd_dim=192,
                 two_emb_layer=False):
        super(EFCRes2Net, self).__init__() # Use new class name
        input_size = max(input_size, 1); m_channels = max(m_channels, 1); embd_dim = max(embd_dim, 1)

        self.in_planes = m_channels
        self.expansion = expansion
        self.embd_dim = embd_dim
        self.two_emb_layer = two_emb_layer

        freq_dim_after_conv = math.ceil(input_size / 8.0)
        final_conv_channels = m_channels * 8 * self.expansion
        self.stats_dim_actual = max(int(2 * final_conv_channels * freq_dim_after_conv), 1)

        # Use DepthwiseSeparableConv for initial conv
        self.conv1 = DepthwiseSeparableConv(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels); self.relu1 = ReLU(inplace=True)

        layer_out_channels = []
        # Use the 'block' (Res2NetBlock) passed in __init__
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)
        # Use the 'block' (Res2NetBlock) passed in __init__
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)
        # Use the 'block_fuse' (EFCFusionRes2NetBlock) passed in __init__
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)
        # Use the 'block_fuse' (EFCFusionRes2NetBlock) passed in __init__
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, base_width=base_width, scale=scale)
        layer_out_channels.append(self.in_planes)

        l1_out_c, l2_out_c, l3_out_c, l4_out_c = layer_out_channels
        ds1_out_c = max(l2_out_c, 1)
        # Use DepthwiseSeparableConv for downsampling
        self.layer1_downsample = nn.Sequential(
            DepthwiseSeparableConv(max(l1_out_c, 1), ds1_out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ds1_out_c), ReLU(inplace=True)
        ) if l1_out_c > 0 else nn.Identity()
        self.fuse_mode12 = EFC(c1=ds1_out_c, c2=ds1_out_c) if ds1_out_c > 0 else nn.Identity()

        ds2_out_c = max(l3_out_c, 1)
        self.layer2_downsample = nn.Sequential(
            DepthwiseSeparableConv(ds1_out_c, ds2_out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ds2_out_c), ReLU(inplace=True)
        ) if ds1_out_c > 0 else nn.Identity()
        self.fuse_mode123 = EFC(c1=ds2_out_c, c2=ds2_out_c) if ds2_out_c > 0 else nn.Identity()

        ds3_out_c = max(l4_out_c, 1)
        self.layer3_downsample = nn.Sequential(
            DepthwiseSeparableConv(ds2_out_c, ds3_out_c, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(ds3_out_c), ReLU(inplace=True)
        ) if ds2_out_c > 0 else nn.Identity()
        self.fuse_mode1234 = EFC(c1=ds3_out_c, c2=ds3_out_c) if ds3_out_c > 0 else nn.Identity()

        self.gam = GAM_Attention(in_channels=ds3_out_c) if ds3_out_c > 0 else nn.Identity()
        self.pooling = TemporalStatsPool()

        self.seg_1 = nn.Linear(self.stats_dim_actual, embd_dim)
        if self.two_emb_layer:
            self.relu_emb = nn.ReLU(inplace=True)
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.relu_emb = nn.Identity(); self.seg_bn_1 = nn.Identity(); self.seg_2 = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                except ValueError: nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                 if m.affine: nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        # This method now receives the renamed block classes (Res2NetBlock or EFCFusionRes2NetBlock)
        strides = [stride] + [1] * (num_blocks - 1); layers = []
        current_planes = max(planes, 1)
        try:
            # Instantiates the passed block class (e.g., Res2NetBlock)
            layers.append(block(self.expansion, self.in_planes, current_planes, stride=strides[0], base_width=base_width, scale=scale))
            self.in_planes = current_planes * self.expansion
        except Exception as e: print(f"Error creating first block: {e}"); layers.append(nn.Identity())
        for i in range(1, num_blocks):
            try:
                if self.in_planes > 0:
                    # Instantiates the passed block class
                    layers.append(block(self.expansion, self.in_planes, current_planes, stride=strides[i], base_width=base_width, scale=scale))
                else: layers.append(nn.Identity())
            except Exception as e: print(f"Error creating block {i}: {e}"); layers.append(nn.Identity())
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() != 3: raise ValueError(f"Expected 3D input (B, T, F), got {x.dim()}")
        B, T, F = x.shape
        if F == 0 or T == 0: return torch.zeros(B, self.embd_dim, device=x.device)
        x = x.permute(0, 2, 1).unsqueeze(1) # (B, 1, F, T)

        out = self.conv1(x); out = self.bn1(out); out = self.relu1(out) # Initial DSConv
        out1 = self.layer1(out); out2 = self.layer2(out1); out3 = self.layer3(out2); out4 = self.layer4(out3) # Layers using renamed blocks

        out1_ds = self.layer1_downsample(out1) # DSConv Downsample
        if isinstance(self.fuse_mode12, EFC) and out2.shape == out1_ds.shape: fuse_out12 = self.fuse_mode12(out2, out1_ds)
        else: fuse_out12 = out2

        fuse_out12_ds = self.layer2_downsample(fuse_out12) # DSConv Downsample
        if isinstance(self.fuse_mode123, EFC) and out3.shape == fuse_out12_ds.shape: fuse_out123 = self.fuse_mode123(out3, fuse_out12_ds)
        else: fuse_out123 = out3

        fuse_out123_ds = self.layer3_downsample(fuse_out123) # DSConv Downsample
        if isinstance(self.fuse_mode1234, EFC) and out4.shape == fuse_out123_ds.shape: fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_ds)
        else: fuse_out1234 = out4

        if isinstance(self.gam, GAM_Attention): att_out = self.gam(fuse_out1234) # GAM using DSConv
        else: att_out = fuse_out1234

        if att_out.numel() == 0 or att_out.shape[2] == 0 or att_out.shape[3] == 0:
            return torch.zeros(B, self.embd_dim, device=x.device)

        stats = self.pooling(att_out)
        if stats.shape[1] != self.stats_dim_actual:
             raise ValueError(f"Dim mismatch seg_1: Expected {self.stats_dim_actual}, got {stats.shape[1]}. Check input_size ({F}).")

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out_emb = self.relu_emb(embed_a); out_emb = self.seg_bn_1(out_emb); embed_b = self.seg_2(out_emb)
            return embed_b
        else:
            return embed_a

# --- 主函数 ---
if __name__ == "__main__":
    # 1. 定义模型和输入超参数
    input_feat_dim = 80
    time_steps = 98
    batch_size = 1
    embedding_dim = 192
    m_channels_val = 32

    # 2. 实例化模型 (使用新名称 EFCRes2Net)
    print("正在实例化 EFCRes2Net 模型 (使用深度可分离卷积)...")
    try:
        # Instantiate using the new class name
        model = EFCRes2Net(input_size=input_feat_dim,
                           embd_dim=embedding_dim,
                           m_channels=m_channels_val,
                           two_emb_layer=False)
        print("模型实例化成功！")
    except Exception as e:
        print(f"模型实例化失败: {e}")
        exit()

    # 3. 使用 torchsummary 打印模型摘要
    print("-" * 80)
    if _torchsummary_available:
        print("使用 torchsummary 生成模型摘要:")
        try:
            # Input shape for torchsummary: (channels, freq_dim, time_steps)
            input_shape_for_summary = (1, input_feat_dim, time_steps)
            print(f"提供给 torchsummary 的 input_size: {input_shape_for_summary}")
            # Use the aliased import
            torchsummary_summary(model, input_size=input_shape_for_summary, device="cpu")
        except Exception as e:
            print(f"\n运行 torchsummary summary 时出错: {e}")
            # Fallback 参数计数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("\n基本参数计数:")
            print(f"  Total params: {total_params:,}")
            print(f"  Trainable params: {trainable_params:,}")
            print(f"  Non-trainable params: {total_params - trainable_params:,}")
    else:
        # 如果 torchsummary 不可用，打印基本计数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("torchsummary 未安装，仅显示基本参数计数:")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Non-trainable params: {total_params - trainable_params:,}")
    print("-" * 80)
