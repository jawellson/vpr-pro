import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvector.models.pooling import TemporalAveragePooling, TemporalStatsPool, AttentiveStatisticsPooling


# --- ReLU ---
class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'

# ---  Depthwise Separable Conv ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        in_channels = max(1, in_channels)
        out_channels = max(1, out_channels)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
# --- Channel Gate Module ---
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        channels = max(1, channels)
        reduced_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.eps = 1e-10

    def forward(self, x):
        b, c, _, _ = x.size()
        if c == 0: return x
        y = self.avg_pool(x)
        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        y = self.fc(y)
        y = y.clamp(min=self.eps, max=1.0 - self.eps)
        return x * y.expand_as(x)

# --- Enhanced Depthwise Separable Convolution Module ---
class EnhancedDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction_ratio=16, use_gate=True):
        super().__init__()
        in_channels = max(1, in_channels)
        out_channels = max(1, out_channels)
        self.ds_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size,
                                              stride=stride, padding=padding, bias=bias)
        self.use_gate = use_gate
        if self.use_gate and out_channels > 0:
            self.gate = ChannelGate(out_channels, reduction_ratio)
        else:
            self.gate = nn.Identity()

    def forward(self, x):
        x = self.ds_conv(x)
        x = self.gate(x)
        return x

# --- conv1x1 ---
def conv1x1(in_planes, out_planes, stride=1):
    in_planes = max(1, in_planes)
    out_planes = max(1, out_planes)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

# --- conv3x3 ---
def conv3x3(in_planes, out_planes, stride=1, use_gate=True, reduction_ratio=16):
    in_planes = max(1, in_planes)
    out_planes = max(1, out_planes)
    # Use EnhancedDSConv
    return EnhancedDSConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                          use_gate=use_gate, reduction_ratio=reduction_ratio)

# --- EFC Module  ---
class EFC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Ensure the channel is effective
        c1 = max(1, c1)
        c2 = max(1, c2)
        # Calculate the interaction channels
        interact_channels = max(c2 // 4, 1)

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm2d(c2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn_dwconv = nn.BatchNorm2d(c2)
        self.bn_conv3 = nn.BatchNorm2d(c2)
        self.bn_conv4 = nn.BatchNorm2d(c2)

        self.sigmoid = nn.Sigmoid()


        self.group_num = 16 if c2 >= 16 else max(1, c2)

        self.eps = 1e-10
        # Group Normalization parameters
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1) * 0.1)
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))

        # Gate generator
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            ReLU(True),
            nn.Softmax(dim=1),
        )

        # Deep convolution for low-frequency paths
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False)

        # Adaptive pooling for reweighting calculations
        self.Apt = nn.AdaptiveAvgPool2d(1)

        # Pointwise convolution for high-frequency paths
        self.conv4_global = nn.Conv2d(c2, 1, kernel_size=1, stride=1)

        # Intra-group interactions
        self.interacts = nn.ModuleList([
            DepthwiseSeparableConv(interact_channels, interact_channels, kernel_size=1, stride=1, bias=False)
            for _ in range(4)
        ]) if interact_channels > 0 else nn.ModuleList()

        self.valid_split = (c2 > 0 and c2 % 4 == 0 and len(self.interacts) == 4)

    def forward(self, x1, x2):
       # Input projection and initial composition
        global_conv1 = self.conv1(x1)
        bn_x1 = self.bn1(global_conv1)

        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn2(global_conv2)

        # Combine using BN outputs
        X_GLOBAL = bn_x1 + bn_x2

        # Calculate weights based on BN outputs
        weight_1 = self.sigmoid(bn_x1).clamp(min=self.eps, max=1.0 - self.eps)
        weight_2 = self.sigmoid(bn_x2).clamp(min=self.eps, max=1.0 - self.eps)

        # Global feature processing
        x_conv4_w = self.conv4_global(X_GLOBAL)
        X_4_sigmoid = self.sigmoid(x_conv4_w).clamp(min=self.eps, max=1.0 - self.eps)
        X_ = X_4_sigmoid * X_GLOBAL

        # Handle group interactions
        if self.valid_split:
            X_chunks = X_.chunk(4, dim=1)
            out_interact = []
            for group_id in range(4):
                interact = self.interacts[group_id]
               # -- Target specific normalization in the interaction --
                out_chunk = interact(X_chunks[group_id])
                N, C_chunk, H, W = out_chunk.size()
                if N > 0 and C_chunk > 0 and H > 0 and W > 0:
                    x_1_map = out_chunk.view(N, 1, -1)
                    mean_1 = x_1_map.mean(dim=2, keepdim=True).clamp(min=self.eps)
                    x_1_av = x_1_map / mean_1
                    x_2_2 = F.softmax(x_1_av, dim=2)
                    x1_norm = x_2_2.view(N, C_chunk, H, W)

                    x1 = X_chunks[group_id] * x1_norm
                    out_interact.append(x1)
                else:
                    out_interact.append(X_chunks[group_id])
            if out_interact:
                 out = torch.cat(out_interact, dim=1)
            else:
                 out = X_
        else:
            out = X_

        # Normalization
        N, C, H, W = out.size()
        # Check divisibility and non-empty tensors for normalization
        if N > 0 and C > 0 and self.group_num > 0 and C % self.group_num == 0:
             x_add_1 = out.view(N, self.group_num, -1)
             if X_GLOBAL.size() == out.size():
                 x_shape_1 = X_GLOBAL.view(N, self.group_num, -1)
                 mean_1 = x_shape_1.mean(dim=2, keepdim=True).clamp(min=-1e5, max=1e5)
                 std_1 = x_shape_1.std(dim=2, keepdim=True).clamp(min=self.eps)

                 x_guiyi = (x_add_1 - mean_1) / std_1
                 x_guiyi_1 = x_guiyi.view(N, C, H, W)
                 if self.gamma.shape[0] == C and self.beta.shape[0] == C:
                      x_gui = (x_guiyi_1 * self.gamma + self.beta).clamp(min=-1e5, max=1e5)
                 else:
                      x_gui = x_guiyi_1
             else:
                 x_gui = out
        else:
             x_gui = out

        # Reweighting based on global average pooling
        weight_x3 = self.Apt(X_GLOBAL)
        reweights = self.sigmoid(weight_x3).clamp(min=self.eps, max=1.0 - self.eps)
        # Determine the high/low frequency paths based on the comparison between the reweighted weights and the initial weights
        x_up_1 = (reweights >= weight_1).float()
        x_low_1 = (reweights < weight_1).float()
        x_up_2 = (reweights >= weight_2).float()
        x_low_2 = (reweights < weight_2).float()

        # is the combination feature of high frequency and low frequency paths
        x_up = x_up_1 * X_GLOBAL + x_up_2 * X_GLOBAL
        x_low = x_low_1 * X_GLOBAL + x_low_2 * X_GLOBAL

        # Low frequency path processing (DWConv -> BN -> ReLU -> Conv1x1 -> BN -> gating)
        x_low_dw = self.dwconv(x_low)
        x_low_dw = self.bn_dwconv(x_low_dw)
        x_low_dw = ReLU(True)(x_low_dw)
        x_low_pw = self.conv3(x_low_dw) # Pointwise after DWConv
        x_low_pw = self.bn_conv3(x_low_pw)

        x_gate = self.gate_generator(x_low).clamp(min=self.eps, max=1.0 - self.eps) # 门控值
        x_low_gated = x_low_pw * x_gate

        # High-frequency Path Processing (Conv1x1 -> BN)
        x_up_pw = self.conv4(x_up)
        x_up_pw = self.bn_conv4(x_up_pw)

        # Compose paths
        xL = x_low_gated + x_up_pw
        # Add group normalization
        xL = xL + x_gui
        # Final NaN check
        xL = torch.nan_to_num(xL, nan=0.0, posinf=1e5, neginf=-1e5)
        return xL


# --- GAM_Attention optimization module  ---
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4, reduction_ratio=16):
        super(GAM_Attention, self).__init__()
        in_channels = max(in_channels, 1)
        inter_channels = max(in_channels // rate, 1)

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, inter_channels),
            ReLU(True),
            nn.Linear(inter_channels, in_channels)
        )

        # patial Attention - Use EnhancedDSConv
        self.spatial_attention = nn.Sequential(
            EnhancedDSConv(in_channels, inter_channels, kernel_size=7, padding=3, bias=False, use_gate=True, reduction_ratio=reduction_ratio),
            nn.BatchNorm2d(inter_channels),
            ReLU(True),
            EnhancedDSConv(inter_channels, in_channels, kernel_size=7, padding=3, bias=False, use_gate=True, reduction_ratio=reduction_ratio),
            nn.BatchNorm2d(in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 0: return x # Handle zero-channel case

        # Channel attention path
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c) # (B, H*W, C)

        if x_permute.shape[1] > 0: # Check if spatial dimensions exist
            x_att_permute = self.channel_attention(x_permute)
            x_att_permute = x_att_permute.view(b, h, w, c)
            x_channel_att_raw = x_att_permute.permute(0, 3, 1, 2) # (B, C, H, W)
            channel_att_weight = self.sigmoid(x_channel_att_raw).clamp(min=1e-10, max=1.0 - 1e-10)
        else:
            channel_att_weight = torch.ones_like(x)

        x_channel_weighted = x * channel_att_weight

        spatial_att_features = self.spatial_attention(x_channel_weighted)
        spatial_att_weight = self.sigmoid(spatial_att_features).clamp(min=1e-10, max=1.0 - 1e-10)

        # Apply spatial attention
        out = x_channel_weighted * spatial_att_weight

        out = torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)
        return out

# # --- Res2NetBlock---
class Res2NetBlock(nn.Module):
    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2, use_gate_in_block=True, reduction_ratio=16):
        super(Res2NetBlock, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        conv1_out_planes = max(1, width * scale)
        self.conv1 = conv1x1(in_planes, conv1_out_planes, stride)
        self.bn1 = nn.BatchNorm2d(conv1_out_planes)
        self.nums = scale

        convs = []
        bns = []
        conv_width = max(1, width)
        for i in range(self.nums):
            convs.append(conv3x3(conv_width, conv_width, use_gate=use_gate_in_block, reduction_ratio=reduction_ratio))
            bns.append(nn.BatchNorm2d(conv_width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        conv3_out_planes = max(1, planes * self.expansion)
        self.conv3 = conv1x1(conv1_out_planes, conv3_out_planes)
        self.bn3 = nn.BatchNorm2d(conv3_out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
             in_planes_sc = max(1, in_planes)
             out_planes_sc = max(1, self.expansion * planes)
             self.shortcut = nn.Sequential(
                 nn.Conv2d(in_planes_sc, out_planes_sc, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(out_planes_sc))
        self.stride = stride
        self.width = conv_width
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
                sp = sp + spx[i]
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

# # --- EFCFusionRes2NetBlock---
class EFCFusionRes2NetBlock(nn.Module):
    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2, use_gate_in_block=True, reduction_ratio=16):
        super(EFCFusionRes2NetBlock, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        conv1_out_planes = max(1, width * scale)
        conv_width = max(1, width)
        fuse_model_channels = max(1, width)
        self.conv1 = conv1x1(in_planes, conv1_out_planes, stride)
        self.bn1 = nn.BatchNorm2d(conv1_out_planes)
        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(conv_width, conv_width, use_gate=use_gate_in_block, reduction_ratio=reduction_ratio))
            bns.append(nn.BatchNorm2d(conv_width))
        if fuse_model_channels > 0:
             for j in range(self.nums - 1):
                fuse_models.append(EFC(c1=fuse_model_channels, c2=fuse_model_channels))
        else:
             for j in range(self.nums - 1):
                 fuse_models.append(nn.Identity())


        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        conv3_out_planes = max(1, planes * self.expansion)
        self.conv3 = conv1x1(conv1_out_planes, conv3_out_planes)
        self.bn3 = nn.BatchNorm2d(conv3_out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            in_planes_sc = max(1, in_planes)
            out_planes_sc = max(1, self.expansion * planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes_sc, out_planes_sc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes_sc))
        self.stride = stride
        self.width = conv_width
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

# --- EFCRes2Net_GAM  ---
class EFCRes2Net_GAM(nn.Module):
    def __init__(self,
                 input_size,
                 block=Res2NetBlock,
                 block_fuse=EFCFusionRes2NetBlock,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 mul_channel=1,
                 expansion=2,
                 base_width=32,
                 scale=2,
                 embd_dim=192,
                 two_emb_layer=False,
                 reduction_ratio=16,
                 use_gate_in_downsample=True,
                 use_gate_in_conv1=True):
        super(EFCRes2Net_GAM, self).__init__()

        # --- Parameter validation ---
        input_size = max(1, input_size)
        m_channels = max(1, m_channels)
        embd_dim = max(1, embd_dim)
        mul_channel = max(1, mul_channel)
        base_width = max(1, base_width)
        scale = max(1, scale)
        expansion = max(1, expansion)
        reduction_ratio = max(1, reduction_ratio)

        self.in_planes = m_channels
        self.expansion = expansion
        self.embd_dim = embd_dim
        self.two_emb_layer = two_emb_layer
        self.mul_channel = mul_channel
        self.n_stats = 2


        # --- Calculate dimensions ---
        freq_dim_after_conv1 = input_size
        freq_dim_after_layer1 = freq_dim_after_conv1
        freq_dim_after_layer2 = math.ceil(freq_dim_after_layer1 / 2.0)
        freq_dim_after_layer3 = math.ceil(freq_dim_after_layer2 / 2.0)
        freq_dim_after_layer4 = math.ceil(freq_dim_after_layer3 / 2.0)

        channels_after_layer4 = max(1, m_channels * 8 * self.expansion)
        # Downsampling targets:
        ds1_target_out_c = max(1, m_channels * 2 * self.expansion * self.mul_channel)
        # Let's redefine target channels based on the layer outputs they fuse with
        l1_out_c = max(1, m_channels * self.expansion)
        l2_out_c = max(1, m_channels * 2 * self.expansion)
        l3_out_c = max(1, m_channels * 4 * self.expansion)
        l4_out_c = max(1, m_channels * 8 * self.expansion)

        # Channels for downsampling layers to match fusion inputs
        # ds1 output fuses with l2 output (l2_out_c)
        ds1_target_out_c = l2_out_c * self.mul_channel
        # ds2 output fuses with l3 output (l3_out_c)
        ds2_in_c = ds1_target_out_c # Output of fuse12
        ds2_target_out_c = l3_out_c * self.mul_channel
        # ds3 output fuses with l4 output (l4_out_c)
        ds3_in_c = ds2_target_out_c # Output of fuse123
        ds3_target_out_c = l4_out_c * self.mul_channel

        # Recalculate channels_before_pooling based on the final fusion output
        channels_before_pooling = ds3_target_out_c
        # Final spatial dimension before pooling (output of layer4, or fuse_out1234)
        freq_dim_final = freq_dim_after_layer4

        # Stats dim calculation
        self.stats_dim = max(1, int(freq_dim_final * channels_before_pooling))


        # --- Initial Convolution ---
        self.conv1 = EnhancedDSConv(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                    use_gate=use_gate_in_conv1, reduction_ratio=reduction_ratio)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.relu1 = ReLU(inplace=True)

        # --- Backbone Layers ---
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1, base_width=base_width, scale=scale, use_gate_in_block=True, reduction_ratio=reduction_ratio)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2, base_width=base_width, scale=scale, use_gate_in_block=True, reduction_ratio=reduction_ratio)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2, base_width=base_width, scale=scale, use_gate_in_block=True, reduction_ratio=reduction_ratio)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2, base_width=base_width, scale=scale, use_gate_in_block=True, reduction_ratio=reduction_ratio)

        # --- Downsampling Layers  ---
        l1_actual_out_c = l1_out_c

        # Layer 1 Downsample
        self.layer1_downsample = nn.Sequential(
                EnhancedDSConv(l1_actual_out_c, ds1_target_out_c, kernel_size=3, padding=1, stride=2, bias=False, use_gate=use_gate_in_downsample, reduction_ratio=reduction_ratio),
                nn.BatchNorm2d(ds1_target_out_c), ReLU(True)
        ) if l1_actual_out_c > 0 and ds1_target_out_c > 0 else nn.Identity()

        # Layer 2 Downsample
        self.layer2_downsample = nn.Sequential(
                EnhancedDSConv(ds1_target_out_c, ds2_target_out_c, kernel_size=3, padding=1, stride=2, bias=False, use_gate=use_gate_in_downsample, reduction_ratio=reduction_ratio),
                nn.BatchNorm2d(ds2_target_out_c), ReLU(True)
        ) if ds1_target_out_c > 0 and ds2_target_out_c > 0 else nn.Identity()

        # Layer 3 Downsample
        self.layer3_downsample = nn.Sequential(
                EnhancedDSConv(ds2_target_out_c, ds3_target_out_c, kernel_size=3, padding=1, stride=2, bias=False, use_gate=use_gate_in_downsample, reduction_ratio=reduction_ratio),
                nn.BatchNorm2d(ds3_target_out_c), ReLU(True)
        ) if ds2_target_out_c > 0 and ds3_target_out_c > 0 else nn.Identity()


        # --- Fusion Modules ---
        fuse12_channels = ds1_target_out_c # Channel dim after downsample/fusion target
        self.fuse_mode12 = EFC(c1=fuse12_channels, c2=fuse12_channels) if fuse12_channels > 0 else nn.Identity()

        # fuse_mode123 fuses Layer 3 output (l3_out_c) and downsampled fuse12 output (ds2_target_out_c)
        fuse123_channels = ds2_target_out_c
        self.fuse_mode123 = EFC(c1=fuse123_channels, c2=fuse123_channels) if fuse123_channels > 0 else nn.Identity()

        # fuse_mode1234 fuses Layer 4 output (l4_out_c) and downsampled fuse123 output (ds3_target_out_c)
        fuse1234_channels = ds3_target_out_c
        self.fuse_mode1234 = EFC(c1=fuse1234_channels, c2=fuse1234_channels) if fuse1234_channels > 0 else nn.Identity()

        # --- Attention Module ---
        gam_in_channels = fuse1234_channels
        self.gam = GAM_Attention(in_channels=gam_in_channels, reduction_ratio=reduction_ratio) if gam_in_channels > 0 else nn.Identity()

        # --- Pooling ---
        self.pooling = TemporalStatsPool()

        # --- Embedding Layers ---
        linear_input_dim = max(1, gam_in_channels * freq_dim_final * self.n_stats)
        self.stats_dim = max(1, gam_in_channels * freq_dim_final)

        self.seg_1 = nn.Linear(linear_input_dim, embd_dim)

        if self.two_emb_layer:
            self.relu_emb = ReLU(True)
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False) if embd_dim > 1 else nn.Identity()
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.relu_emb = nn.Identity()
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale, use_gate_in_block=True, reduction_ratio=16):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_planes = max(1, planes)
        for current_stride in strides:
             if self.in_planes <= 0:
                layers.append(nn.Identity())
                self.in_planes = 1
             else:
                 layers.append(block(self.expansion, self.in_planes, current_planes, current_stride, base_width, scale,
                                     use_gate_in_block=use_gate_in_block, reduction_ratio=reduction_ratio))
                 self.in_planes = max(1, current_planes * self.expansion)

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, DepthwiseSeparableConv):
                     if hasattr(m, 'pointwise') and isinstance(m.pointwise, nn.Conv2d):
                         nn.init.kaiming_normal_(m.pointwise.weight, mode='fan_out', nonlinearity='relu')
                         if m.pointwise.bias is not None:
                             nn.init.constant_(m.pointwise.bias, 0)
                     if hasattr(m, 'depthwise') and isinstance(m.depthwise, nn.Conv2d):
                         nn.init.xavier_normal_(m.depthwise.weight)
                         if m.depthwise.bias is not None:
                             nn.init.constant_(m.depthwise.bias, 0)
                elif isinstance(m, EnhancedDSConv):
                     pass
                elif not isinstance(m.modules(), (EnhancedDSConv, DepthwiseSeparableConv)):
                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                     if m.bias is not None:
                         nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                 if m.affine:
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
            # elif isinstance(m, ChannelGate):
            #      for layer in m.fc:
            #          if isinstance(layer, nn.Conv2d):
            #              nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') # Or Xavier
            #              if layer.bias is not None:
            #                  nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        # --- Input Validation ---
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got {x.dim()}D shape {x.shape}")
        B, T, F = x.shape
        if F == 0 or T == 0:
            # print("Input tensor T or F dimension is zero")
            return torch.zeros(B, self.embd_dim, device=x.device, dtype=x.dtype)

        # Permute and unsqueeze: (B, T, F) => (B, F, T) -> (B, 1, F, T)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # --- Backbone Layers ---
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # --- Fusion Path ---

        # 1. Downsample layer 1 output
        out1_downsample = self.layer1_downsample(out1)

        # 2. Fuse layer 2 output with downsampled layer 1 output
        fuse_input1_l2 = out2
        fuse_input2_l1ds = out1_downsample
        if isinstance(self.fuse_mode12, EFC) and fuse_input1_l2.shape[1] == fuse_input2_l1ds.shape[1] and fuse_input1_l2.shape[2:] == fuse_input2_l1ds.shape[2:]:
             fuse_out12 = self.fuse_mode12(fuse_input1_l2, fuse_input2_l1ds)
        elif fuse_input1_l2.shape == fuse_input2_l1ds.shape:
             fuse_out12 = fuse_input1_l2 + fuse_input2_l1ds
        else:
             fuse_out12 = fuse_input1_l2

        # 3. Downsample fused 1&2 output
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)

        # 4. Fuse layer 3 output with downsampled fused 1&2 output
        fuse_input1_l3 = out3
        fuse_input2_f12ds = fuse_out12_downsample
        if isinstance(self.fuse_mode123, EFC) and fuse_input1_l3.shape[1] == fuse_input2_f12ds.shape[1] and fuse_input1_l3.shape[2:] == fuse_input2_f12ds.shape[2:]:
            fuse_out123 = self.fuse_mode123(fuse_input1_l3, fuse_input2_f12ds)
        elif fuse_input1_l3.shape == fuse_input2_f12ds.shape:
            fuse_out123 = fuse_input1_l3 + fuse_input2_f12ds
        else:
            fuse_out123 = fuse_input1_l3

        # 5. Downsample fused 1&2&3 output
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)

        # 6. Fuse layer 4 output with downsampled fused 1&2&3 output
        fuse_input1_l4 = out4
        fuse_input2_f123ds = fuse_out123_downsample
        if isinstance(self.fuse_mode1234, EFC) and fuse_input1_l4.shape[1] == fuse_input2_f123ds.shape[1] and fuse_input1_l4.shape[2:] == fuse_input2_f123ds.shape[2:]:
             fuse_out1234 = self.fuse_mode1234(fuse_input1_l4, fuse_input2_f123ds)
        elif fuse_input1_l4.shape == fuse_input2_f123ds.shape:
            fuse_out1234 = fuse_input1_l4 + fuse_input2_f123ds
        else:
             fuse_out1234 = fuse_input1_l4

        # --- GAM Attention Optimization ---
        if isinstance(self.gam, GAM_Attention):
             att_out = self.gam(fuse_out1234)
        else:
             att_out = fuse_out1234

        # --- Pooling ---
        if att_out.numel() == 0 or att_out.shape[2] == 0 or att_out.shape[3] == 0:
             return torch.zeros(B, self.embd_dim, device=x.device, dtype=x.dtype)
        if torch.isnan(att_out).any() or torch.isinf(att_out).any():
            att_out = torch.nan_to_num(att_out, nan=0.0)
        stats = self.pooling(att_out)

        # --- Embedding Layers ---
        if torch.isnan(stats).any() or torch.isinf(stats).any():
             stats = torch.nan_to_num(stats, nan=0.0)

        embed_a = self.seg_1(stats)

        if self.two_emb_layer:
            out_emb = self.relu_emb(embed_a)
            if isinstance(self.seg_bn_1, nn.BatchNorm1d) and out_emb.shape[0] > 1:
                 out_emb = self.seg_bn_1(out_emb)
            elif isinstance(self.seg_bn_1, nn.BatchNorm1d) and out_emb.shape[0] <= 1:
                 pass

            if torch.isnan(out_emb).any() or torch.isinf(out_emb).any():
                 out_emb = torch.nan_to_num(out_emb, nan=0.0)
            embed_b = self.seg_2(out_emb)
            embed_b = torch.nan_to_num(embed_b, nan=0.0)
            return embed_b
        else:
            embed_a = torch.nan_to_num(embed_a, nan=0.0)
            return embed_a
