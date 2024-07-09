#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()

# TODO: add dice loss


def get_loss(output_image: torch.Tensor, lable_image: torch.Tensor) -> torch.Tensor:
    return loss_function(output_image, lable_image.float())
