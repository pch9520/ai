#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()


def get_loss(output_image: torch.Tensor, lable_image: torch.Tensor) -> torch.Tensor:
    print(output_image.shape)
    print(lable_image.shape)
    return loss_function(output_image, lable_image)
