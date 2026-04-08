import torch


def dice(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + 1) / (pred.sum() + target.sum() + 1)
