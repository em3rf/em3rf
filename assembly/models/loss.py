import torch


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))
    return loss.mean()
