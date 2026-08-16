import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(Loss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Flatten tensors
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        # 1. Pixel-wise Binary Cross Entropy (BCE)
        # Using _with_logits is numerically stable as it applies sigmoid internally
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_flat, 
            targets_flat, 
            reduction='mean'
        )

        # 2. Soft Dice Loss 
        # Directly optimizes for spatial overlap (Intersection over Union approximation)
        probs = torch.sigmoid(logits_flat)
        
        intersection = (probs * targets_flat).sum()
        union = probs.sum() + targets_flat.sum()
        
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth))

        # Combined Loss
        return bce_loss + dice_loss