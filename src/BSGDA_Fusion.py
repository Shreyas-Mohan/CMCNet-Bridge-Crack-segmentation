import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Performs fast 1D convolution across channels without dimensionality reduction.
    Kernel size k is dynamically computed based on channel dimensions.
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 != 0 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.view(b, 1, c))  # [B, 1, C]
        y = self.sigmoid(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        return x * y


class BSGDA_Fusion(nn.Module):
    """
    Bidirectional Semantic-Decoupled Aggregation (BS-GDA) Module.
    Designed for fine-grained binary crack segmentation under dense concrete noise.
    
    Incorporates:
    1. Joint mutual spatial pre-filtering.
    2. Defect saliency gating (A_crack) and explicit background suppression (bg):
       W_clean = A_crack * (1.0 - bg).
    3. Decoupled cross-branch feature aggregation with learnable scaling (gamma).
    4. Efficient Channel Attention (ECA) recalibration.
    """
    def __init__(self, in_channel: int, out_channel: int = None):
        super(BSGDA_Fusion, self).__init__()
        self.dim = in_channel
        out_channel = out_channel if out_channel is not None else in_channel
        self.out_channel = out_channel

        # 1. Mutual Pre-filtering Gate
        self.pre_gate = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 2. Defect Saliency Gate (A_crack)
        self.crack_gate = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 3. Background Texture Suppression Gate (bg)
        self.bg_gate = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // 2),
            nn.GELU(),
            nn.Conv2d(in_channel // 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 4. Decoupled Aggregation Projection
        self.decoupled_proj = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        # Learnable residual modulation factor (initialized to 0.5)
        self.gamma = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        # 5. Channel Refinement and Final Projection
        self.eca = ECABlock(in_channel)
        self.final_proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.adjust_residual = nn.Identity() if in_channel == out_channel else nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x_c: torch.Tensor, x_m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_c: Tensor [B, C, H, W] from DO-Conv CNN branch (Local detail)
            x_m: Tensor [B, C, H, W] from Visual Mamba branch (Global context)
        Returns:
            f_out: Tensor [B, C, H, W] fused feature map
        """
        assert x_c.shape == x_m.shape, f"Dimension mismatch: x_c={x_c.shape}, x_m={x_m.shape}"

        # Step 1: Mutual Pre-filtering
        joint = torch.cat([x_c, x_m], dim=1)  # [B, 2C, H, W]
        spatial_filter = self.pre_gate(joint)  # [B, 1, H, W]
        x_c_gated = x_c * spatial_filter
        x_m_gated = x_m * spatial_filter

        # Step 2: Extract Saliency and Background Suppression Maps
        gated_joint = torch.cat([x_c_gated, x_m_gated], dim=1)
        A_crack = self.crack_gate(gated_joint)  # [B, 1, H, W]
        bg = self.bg_gate(gated_joint)          # [B, 1, H, W]

        # Step 3: Compute Purified Defect Weight
        W_clean = A_crack * (1.0 - bg)  # [B, 1, H, W]

        # Step 4: Extract Decoupled Representations
        f_dec_c = x_c_gated * W_clean
        f_dec_m = x_m_gated * W_clean
        f_cd = self.decoupled_proj(torch.cat([f_dec_c, f_dec_m], dim=1))  # [B, C, H, W]

        # Step 5: Feature Reintegration with ECA
        f_combined = x_c + (self.gamma * f_cd) + x_m_gated
        f_refined = self.eca(f_combined)
        f_out = self.final_proj(f_refined) + self.adjust_residual(x_c)  # Preserve gradient path to CNN

        return f_out
