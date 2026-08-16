import torch.nn as nn
import torch
import torch.nn.functional as F

# Ensure DOConv2d is properly imported from your project structure
from src.Doconv import DOConv2d

class Mambabranch(nn.Module):
    def __init__(self, inchannel):
        super(Mambabranch, self).__init__()
        self.conv1 = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3)
        self.gmp = nn.AdaptiveMaxPool2d(output_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gmp(x1)
        x1 = self.sigmoid(x1)
        out = x1 * x
        return out

class Cnnbranch(nn.Module):
    def __init__(self, inchannel):
        super(Cnnbranch, self).__init__()
        self.conv1 = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=(3, 1))
        self.conv2 = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=(1, 3))
        self.proj = DOConv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = F.max_pool2d(x, kernel_size=(1, W), stride=(1, W))
        x2 = F.max_pool2d(x, kernel_size=(H, 1), stride=(H, 1))
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=True)
        fusion = x1 + x2
        fusion = self.proj(fusion)
        fusion = torch.sigmoid(fusion)
        out = x * fusion
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.adjust_channels = (in_channels != out_channels)
        if self.adjust_channels:
            self.conv_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.adjust_channels:
            identity = self.conv_adjust(identity)
        out += identity
        out = F.relu(out)
        return out

class fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        # 1. Keep the original CNN and Mamba attention blocks
        self.mm = Mambabranch(inchannel=in_channel)
        self.cnn = Cnnbranch(inchannel=in_channel)
        
        # 2. Add the Bidirectional Spatial Cross-Gating path
        hidden_dim = max(in_channel // 4, 1)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channel * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 3. ResidualBlock handles the final projection to out_channel
        self.residual = ResidualBlock(in_channel * 3, out_channel)

    def forward(self, x1, x2):
        # x1 is CNN branch, x2 is Mamba branch
        x1_attn = self.cnn(x1)
        x2_attn = self.mm(x2)
        
        # Compute joint spatial importance map
        cat_feat = torch.cat([x1_attn, x2_attn], dim=1)
        gate = self.spatial_gate(cat_feat)
        
        # Residual spatial modulation (1 + gate boosts confidence without erasing weak signals)
        cnn_guided = x1_attn * (1.0 + gate)
        mamba_guided = x2_attn * (1.0 + gate)
        
        # Element-wise cross-correlation
        interaction = cnn_guided * mamba_guided
        
        # Final aggregation
        fused = torch.cat([cnn_guided, mamba_guided, interaction], dim=1)
        output = self.residual(fused)
        
        return output