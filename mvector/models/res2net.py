# import math

# import torch
# import torch.nn as nn

# from mvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
# from mvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling


# class Bottle2neck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
#         """ Constructor
#         Args:
#             inplanes: input channel dimensionality
#             planes: output channel dimensionality
#             stride: conv stride. Replaces pooling layer.
#             downsample: None when stride = 1
#             baseWidth: basic width of conv3x3
#             scale: number of scale.
#             type: 'normal': normal set. 'stage': first block of a new stage.
#         """
#         super(Bottle2neck, self).__init__()

#         width = int(math.floor(planes * (baseWidth / 64.0)))
#         self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width * scale)

#         if scale == 1:
#             self.nums = 1
#         else:
#             self.nums = scale - 1
#         if stype == 'stage':
#             self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
#         convs = []
#         bns = []
#         for i in range(self.nums):
#             convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
#             bns.append(nn.BatchNorm2d(width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)

#         self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stype = stype
#         self.scale = scale
#         self.width = width

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         spx = torch.split(out, self.width, 1)
#         for i in range(self.nums):
#             if i == 0 or self.stype == 'stage':
#                 sp = spx[i]
#             else:
#                 sp = sp + spx[i]
#             sp = self.convs[i](sp)
#             sp = self.relu(self.bns[i](sp))
#             if i == 0:
#                 out = sp
#             else:
#                 out = torch.cat((out, sp), 1)
#         if self.scale != 1 and self.stype == 'normal':
#             out = torch.cat((out, spx[self.nums]), 1)
#         elif self.scale != 1 and self.stype == 'stage':
#             out = torch.cat((out, self.pool(spx[self.nums])), 1)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Res2Net(nn.Module):

#     def __init__(self, input_size, m_channels=32, layers=[3, 4, 6, 3], base_width=32, scale=2, embd_dim=192,
#                  pooling_type="ASP"):
#         super(Res2Net, self).__init__()
#         self.inplanes = m_channels
#         self.base_width = base_width
#         self.scale = scale
#         self.embd_dim = embd_dim
#         self.conv1 = nn.Conv2d(1, m_channels, kernel_size=7, stride=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(m_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(Bottle2neck, m_channels, layers[0])
#         self.layer2 = self._make_layer(Bottle2neck, m_channels*2, layers[1], stride=2)
#         self.layer3 = self._make_layer(Bottle2neck, m_channels * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(Bottle2neck, m_channels * 8, layers[3], stride=2)

#         if input_size < 96:
#             cat_channels = m_channels * 8 * Bottle2neck.expansion * (input_size // self.base_width)
#         else:
#             cat_channels = m_channels * 8 * Bottle2neck.expansion * (
#                         input_size // self.base_width - int(math.sqrt(input_size / 64)))
#         if pooling_type == "ASP":
#             self.pooling = AttentiveStatisticsPooling(cat_channels, attention_channels=128)
#             self.bn2 = nn.BatchNorm1d(cat_channels * 2)
#             self.linear = nn.Linear(cat_channels * 2, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         elif pooling_type == "SAP":
#             self.pooling = SelfAttentivePooling(cat_channels, 128)
#             self.bn2 = nn.BatchNorm1d(cat_channels)
#             self.linear = nn.Linear(cat_channels, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         elif pooling_type == "TAP":
#             self.pooling = TemporalAveragePooling()
#             self.bn2 = nn.BatchNorm1d(cat_channels)
#             self.linear = nn.Linear(cat_channels, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         elif pooling_type == "TSP":
#             self.pooling = TemporalStatisticsPooling()
#             self.bn2 = nn.BatchNorm1d(cat_channels * 2)
#             self.linear = nn.Linear(cat_channels * 2, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         else:
#             raise Exception(f'没有{pooling_type}池化层！')

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = [block(self.inplanes, planes, stride, downsample=downsample,
#                         stype='stage', baseWidth=self.base_width, scale=self.scale)]
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.transpose(2, 1)
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = x.reshape(x.shape[0], -1, x.shape[-1])
#         print(x.shape)
#         x = self.pooling(x)
#         print(x.shape)
#         x = self.bn2(x)
#         x = self.linear(x)
#         x = self.bn3(x)

#         return x
import math
import torch
import torch.nn as nn

# Assuming these pooling layers are correctly implemented and importable
# If not, you might need to provide dummy implementations for profiling
try:
    from mvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
    from mvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
except ImportError:
    print("Warning: Pooling layers not found. Using dummy implementations for profiling.")
    # Simple dummy implementations if the real ones aren't available
    class DummyPooling(nn.Module):
        def __init__(self, input_dim, *args, **kwargs):
            super().__init__()
            self.input_dim = input_dim
        def forward(self, x):
            # Simulate pooling along the time dimension
            # Output shape depends on the specific pooling type
            # This is a placeholder and might not reflect exact dimensions
            batch_size = x.shape[0]
            # Guess output based on class name (crude)
            if "StatisticsPooling" in self.__class__.__name__:
                 # Stats pooling often concatenates mean and std dev
                return torch.randn(batch_size, self.input_dim * 2, device=x.device)
            else: # TAP, SAP
                return torch.randn(batch_size, self.input_dim, device=x.device)

    class AttentiveStatisticsPooling(DummyPooling): pass
    class TemporalAveragePooling(DummyPooling): pass
    class SelfAttentivePooling(DummyPooling): pass
    class TemporalStatisticsPooling(DummyPooling): pass


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, input_size, m_channels=32, layers=[3, 4, 6, 3], base_width=26, scale=4, embd_dim=192, # Adjusted base_width/scale based on common Res2Net configs
                 pooling_type="ASP"):
        super(Res2Net, self).__init__()
        self.inplanes = m_channels
        self.base_width = base_width
        self.scale = scale
        self.embd_dim = embd_dim
        # Adjusted conv1: kernel_size=3, stride=1, padding=1 is more common for audio spectrograms starting smaller
        # Adjusted conv1: kernel_size=7, stride=2, padding=3 might be used too, depends on input
        # Let's keep original for now: kernel_size=7, stride=3, padding=1
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=7, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottle2neck, m_channels, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, m_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, m_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, m_channels * 8, layers[3], stride=2)

        # --- Calculate output feature dimension after convolutions ---
        # This part is tricky as it depends on the exact output shape of layer4
        # We need a dummy forward pass (or manual calculation) to get the shape before pooling
        # Let's estimate based on strides:
        # Input: (B, 1, F, T) -> (B, C1, F', T') after conv1/pool
        # Strides: conv1(3), pool(2), layer2(2), layer3(2), layer4(2)
        # Total stride approx: 3 * 2 * 2 * 2 * 2 = 48 in Time dim? (Check padding effects)
        # Total stride approx in Freq dim is similar but depends more on kernel sizes/padding
        # Let's use a placeholder calculation - a dummy forward pass is more reliable
        # A better way is to pass a dummy tensor through the CNN part first.
        with torch.no_grad():
            dummy_f = torch.randn(1, 1, input_size, 200) # B=1, C=1, F=input_size, T=200(example)
            dummy_f = self.conv1(dummy_f)
            dummy_f = self.bn1(dummy_f)
            dummy_f = self.relu(dummy_f)
            dummy_f = self.max_pool(dummy_f)
            dummy_f = self.layer1(dummy_f)
            dummy_f = self.layer2(dummy_f)
            dummy_f = self.layer3(dummy_f)
            dummy_f = self.layer4(dummy_f)
            # Shape before pooling is (B, C_out, F_out, T_out)
            # Reshape to (B, C_out * F_out, T_out) for pooling
            b, c_out, f_out, t_out = dummy_f.shape
            cat_channels = c_out * f_out
            print(f"--- Intermediate shape before pooling: {dummy_f.shape} -> Reshaped channels: {cat_channels} ---")

        # Original calculation based on input_size / base_width - might be inaccurate
        # if input_size < 96:
        #     cat_channels = m_channels * 8 * Bottle2neck.expansion * (input_size // 32) # Using 32 instead of base_width?
        # else:
        #      cat_channels = m_channels * 8 * Bottle2neck.expansion * (
        #                 input_size // 32 - int(math.sqrt(input_size / 64))) # Using 32?

        # --- Define Pooling and Final Layers ---
        if pooling_type == "ASP":
            # ASP input dim calculation depends on the output shape of layer4
            self.pooling = AttentiveStatisticsPooling(cat_channels, attention_channels=128)
            pooled_dim = cat_channels * 2 # ASP concatenates mean and std dev
            self.bn2 = nn.BatchNorm1d(pooled_dim)
            self.linear = nn.Linear(pooled_dim, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(cat_channels, hidden_size=128) # Assuming hidden_size=128
            pooled_dim = cat_channels
            self.bn2 = nn.BatchNorm1d(pooled_dim)
            self.linear = nn.Linear(pooled_dim, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            pooled_dim = cat_channels
            self.bn2 = nn.BatchNorm1d(pooled_dim)
            self.linear = nn.Linear(pooled_dim, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            pooled_dim = cat_channels * 2 # TSP concatenates mean and std dev
            self.bn2 = nn.BatchNorm1d(pooled_dim)
            self.linear = nn.Linear(pooled_dim, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                 if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                 if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 nn.init.xavier_normal_(m.weight)
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # Make sure downsample stride matches layer stride
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.base_width, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale, stype='normal')) # Use 'normal' stype

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x expected shape: (batch_size, time_steps, feature_dim) e.g. (B, T, F)
        # Example: (16, 300, 80)
        # print("Input shape:", x.shape)
        x = x.transpose(1, 2) # (B, F, T)
        x = x.unsqueeze(1)    # (B, 1, F, T) - Add channel dimension
        # print("Shape after unsqueeze:", x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # print("Shape after conv1/pool:", x.shape)

        x = self.layer1(x)
        # print("Shape after layer1:", x.shape)
        x = self.layer2(x)
        # print("Shape after layer2:", x.shape)
        x = self.layer3(x)
        # print("Shape after layer3:", x.shape)
        x = self.layer4(x)
        # print("Shape after layer4:", x.shape) # Should be (B, C_out, F_out, T_out)

        # Reshape for pooling: (B, C_out * F_out, T_out)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # print("Shape before pooling:", x.shape)

        x = self.pooling(x)
        # print("Shape after pooling:", x.shape) # Should be (B, pooled_dim)

        # Ensure pooling output is 2D (B, Features) before BN1d/Linear
        if x.dim() > 2:
             # Handle cases where pooling might return extra dims (e.g., keep time dim of size 1)
             x = x.squeeze(-1) # Example, adjust if needed

        x = self.bn2(x)
        x = self.linear(x)
        x = self.bn3(x)
        # print("Final output shape:", x.shape) # Should be (B, embd_dim)

        return x

# ================== Main Function for Calculation ==================
from thop import profile

def main():
    # --- Configuration ---
    # These should match the typical usage of your model
    input_size = 80   # Example: Number of Mel filter banks (Feature dimension)
    time_steps = 300  # Example: Number of frames in an utterance
    batch_size = 1    # Use batch size 1 for profiling FLOPs/Params per sample

    m_channels = 32
    layers = [3, 4, 6, 3] # Corresponds to Res2Net-50 structure
    base_width = 26
    scale = 4
    embd_dim = 192
    pooling_type = "ASP" # Choose the pooling type you want to profile

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate Model ---
    print("\nInstantiating model...")
    model = Res2Net(
        input_size=input_size,
        m_channels=m_channels,
        layers=layers,
        base_width=base_width,
        scale=scale,
        embd_dim=embd_dim,
        pooling_type=pooling_type
    ).to(device)
    model.eval() # Set to evaluation mode

    # --- Create Dummy Input ---
    # The model expects (batch_size, time_steps, feature_dim)
    dummy_input = torch.randn(batch_size, time_steps, input_size).to(device)
    print(f"\nCreated dummy input with shape: {dummy_input.shape}")

    # --- Calculate MACs and Parameters ---
    print("Profiling model...")
    # Note: thop calculates MACs (Multiply-Accumulate operations).
    # FLOPs are often considered roughly 2 * MACs for Conv/Linear layers.
    macs, params = profile(model, inputs=(dummy_input,), verbose=False) # Set verbose=True for detailed breakdown

    # --- Print Results ---
    # Convert to millions (M) and giga (G) for readability
    params_m = params / 1e6
    macs_g = macs / 1e9

    print("\n" + "="*30)
    print(" Model Profiling Results")
    print("="*30)
    print(f" Configuration:")
    print(f"  Input Features (input_size): {input_size}")
    print(f"  Time Steps (for profiling): {time_steps}")
    print(f"  M Channels: {m_channels}")
    print(f"  Layers: {layers}")
    print(f"  Base Width: {base_width}")
    print(f"  Scale: {scale}")
    print(f"  Embedding Dim: {embd_dim}")
    print(f"  Pooling Type: {pooling_type}")
    print("-"*30)
    print(f" Total Parameters: {params_m:.2f} M")
    print(f" Total MACs: {macs_g:.2f} G")
    print(f" Estimated FLOPs (≈ 2*MACs): {macs_g * 2:.2f} G")
    print("="*30)

if __name__ == "__main__":
    main()
