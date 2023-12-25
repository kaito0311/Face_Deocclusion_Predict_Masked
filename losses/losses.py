import os

import torch
import numpy as np
from typing import *
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg as vgg


def discriminator_adv_loss(disc_original_output, disc_generated_output):
    # Discriminator must classify correct real/fake image
    # 0 - fake, 1 - real
    B = disc_original_output.size()[0]
    loss_object = torch.nn.BCEWithLogitsLoss()
    disc_original_output = disc_original_output.view(B, -1)
    disc_generated_output = disc_generated_output.view(B, -1)
    real_loss = loss_object(disc_original_output,
                            torch.ones_like(disc_original_output))
    fake_loss = loss_object(disc_generated_output,
                            torch.zeros_like(disc_generated_output))
    return real_loss, fake_loss, real_loss + fake_loss


def generator_adv_loss(disc_generated_output):
    # Generator must generate a real image
    # Assign output of generate to 1 - real
    B = disc_generated_output.size()[0]
    loss_object = torch.nn.BCEWithLogitsLoss()
    disc_generated_output = disc_generated_output.view(B, -1)
    gen_loss = loss_object(disc_generated_output,
                           torch.ones_like(disc_generated_output))
    return gen_loss


def pixel_wise(out_front, target_front):
    # Only for FRONT image
    # target_front is input image
    return torch.mean(torch.abs(out_front - target_front))


def identity_loss(model_extract_feature, feat_rot, out_rot, feat_front, out_front):
    # Get feat of generator output of ROTATION image
    gen_feat_rot, _, _, _, _ = model_extract_feature(out_rot)
    gen_feat_rot = torch.nn.functional.normalize(gen_feat_rot, p=2, dim=1)
    # Get feat of generator output of FRONT image
    gen_feat_front, _, _, _, _ = model_extract_feature(out_front)
    gen_feat_front = torch.nn.functional.normalize(gen_feat_front, p=2, dim=1)
    # Identity loss
    assert len(gen_feat_front.size()) == 2, "Invalid output dimension: {}".format(
        gen_feat_front.size())
    # L2 loss for both FRONT & ROTATION images
    loss_identity_rot = torch.mean(
        torch.sum((gen_feat_rot - feat_rot)**2, dim=1))
    loss_identity_front = torch.mean(
        torch.sum((gen_feat_front - feat_front)**2, dim=1))
    return (loss_identity_rot + loss_identity_front)/2.0


def identity_loss(model_extract_feature, restore_image, ori_image): 
    feature_restore, _, _, _, _ = model_extract_feature(restore_image)
    feature_restore = torch.nn.functional.normalize(feature_restore, p=2, dim=1) 

    feature_ori, _, _, _, _ = model_extract_feature(ori_image)
    feature_ori = torch.nn.functional.normalize(feature_ori, p=2, dim=1) 

    assert len(feature_restore.size()) == 2, "Invalid output dimension: {}".format(
        feature_restore.size())

    loss_identity = torch.mean(torch.sum((feature_restore - feature_ori) ** 2, dim=1))
    return loss_identity 

class SobelLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SobelLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. ' "Supported ones are: ['none', 'mean', 'sum']")

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * sobel_loss(pred, target, reduction=self.reduction)


def sobel_loss(pred, target, reduction='none'):
    batch_size = pred.size()[0]
    channels = pred.size()[1]
    G_X = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]
    G_Y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]
    X_kernel = torch.FloatTensor(G_X).expand(batch_size, channels, 3, 3)
    Y_kernel = torch.FloatTensor(G_Y).expand(batch_size, channels, 3, 3)
    device = pred.device
    X_kernel = X_kernel.to(device)
    Y_kernel = Y_kernel.to(device)
    X_weight = nn.Parameter(data=X_kernel, requires_grad=False)
    Y_weight = nn.Parameter(data=Y_kernel, requires_grad=False)
    train_Gx = torch.nn.functional.conv2d(pred, X_weight, padding=1)
    train_Gy = torch.nn.functional.conv2d(pred, Y_weight, padding=1)
    ground_Gx = torch.nn.functional.conv2d(target, X_weight, padding=1)
    ground_Gy = torch.nn.functional.conv2d(target, Y_weight, padding=1)
    train_img = torch.sqrt(torch.abs(train_Gx)+torch.abs(train_Gy) + 1e-10)
    ground_img = torch.sqrt(torch.abs(ground_Gx)+torch.abs(ground_Gy) + 1e-10)
    # transform  function, three function, f, sqrt(f),pow(f)
    return F.l1_loss(train_img, ground_img, reduction=reduction)
