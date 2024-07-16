#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.cross_entropy_loss_func = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        cls_num = inputs.size(1)
        assert cls_num == len(self.weight), (f"weight size should be equal to the number of classes, "
                                             f"cls_num: {cls_num}, weight size: {len(self.weight)}")
        ce_loss = self.cross_entropy_loss_func(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss
