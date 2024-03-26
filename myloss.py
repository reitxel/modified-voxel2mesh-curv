# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:23:39 2023

@author: rgonzal2
"""
import torch
import torch.nn as nn

class DiceCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        
        ce_loss = self.CE_loss(pred, target)
        
        pred = pred.sigmoid()  # Apply sigmoid to get probabilities, if not already applied
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 2.0 * (intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        dice_ce_loss = dice_loss + ce_loss
        
        return dice_ce_loss

    
class HybridLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(HybridLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        # Cross-Entropy Loss
        ce_loss = nn.BCEWithLogitsLoss()(logits, targets)

        # Dice Loss
        smooth = 1e-5
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)

        # Combine the losses
        total_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss

        return total_loss
