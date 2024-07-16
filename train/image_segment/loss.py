#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch

from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from utils.utils import get_torch_device


def get_loss(output_image: torch.Tensor, lable_image: torch.Tensor, cross_loss_weight: int = 0.5) -> torch.Tensor:
    focal_loss_func = FocalLoss(weight=torch.tensor([1] + [6] * 20, dtype=torch.float32).to(get_torch_device()))
    cross_entropy_loss = focal_loss_func(output_image, lable_image.float())
    dice_loss_func = DiceLoss(reduction="mean")
    dice_loss = dice_loss_func(output_image, lable_image.float())
    return cross_loss_weight * cross_entropy_loss + (1 - cross_loss_weight) * dice_loss
