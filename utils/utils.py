#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import random
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def init_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def weight_init(model: nn.Module):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def pad_and_resize_img(path: str, size: Tuple[int, int] = (256, 256), mode='P'):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new(mode, (temp, temp), 0)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def format_logger(log: logging.Logger, log_file_path: str) -> None:
    logging.basicConfig(level=logging.INFO,
                        filename=log_file_path,
                        filemode="w",
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(lineno)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)
