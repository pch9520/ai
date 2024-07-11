#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from losses.dice_loss import DiceLoss


def get_loss(output_image: torch.Tensor, lable_image: torch.Tensor, cross_loss_weight: int = 0.5) -> torch.Tensor:
    cross_entropy_loss_func = nn.CrossEntropyLoss()
    cross_entropy_loss = cross_entropy_loss_func(output_image, lable_image.float())
    dice_loss_func = DiceLoss(reduction="mean")
    dice_loss = dice_loss_func(output_image, lable_image.float())
    return cross_loss_weight * cross_entropy_loss + (1 - cross_loss_weight) * dice_loss
