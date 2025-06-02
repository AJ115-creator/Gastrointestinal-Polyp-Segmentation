import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Focal loss formula
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()
        FP = (pred * (1-target)).sum()
        FN = ((1-pred) * target).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0, tversky_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        
        return (self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.tversky_weight * tversky_loss) 