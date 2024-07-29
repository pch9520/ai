#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from typing import Dict

import numpy as np
from PIL import Image


class SingleCameraImage:
    def __init__(self, h: int, w: int, file_path: str, channel: str,
                 timestamp: int, is_key_frame: bool):
        self.exist = os.path.exists(file_path)
        self.h = h
        self.w = w
        self.file_path = file_path
        self.channel = channel
        self.timestamp = timestamp
        self.is_key_frame = is_key_frame
        self.image = None

    def get_image_from_file(self, file_path: str) -> Image:
        return Image.open(file_path)

    def pil2array(self) -> np.array:
        return np.array(self.image)

    def array2pil(self, image: np.array) -> Image:
        return Image.fromarray(image.astype(np.uint8)).convert('RGB')


class CameraImages:
    def __init__(self, images: Dict[str, SingleCameraImage]):
        self.front_image = images['CAM_FRONT']
        self.front_left_image = images['CAM_FRONT_LEFT']
        self.front_right_image = images['CAM_FRONT_RIGHT']
        self.back_image = images['CAM_BACK']
        self.back_left_image = images['CAM_BACK_LEFT']
        self.back_right_image = images['CAM_BACK_RIGHT']
