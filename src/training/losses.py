import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        dice = 1 - (2 * inter + 1) / (pred.sum() + target.sum() + 1)

        return bce_loss + dice
