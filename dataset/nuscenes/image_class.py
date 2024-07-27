#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


class SingleCameraImage:
    def __init__(self, image: np.array):
        self.image = image


class CameraImages:
    def __init__(self, images: dict):
        self.front_image = SingleCameraImage(images['CAM_FRONT'])
        self.front_left_image = SingleCameraImage(images['CAM_FRONT_LEFT'])
        self.front_right_image = SingleCameraImage(images['CAM_FRONT_RIGHT'])
        self.back_image = SingleCameraImage(images['CAM_BACK'])
        self.back_left_image = SingleCameraImage(images['CAM_BACK_LEFT'])
        self.back_right_image = SingleCameraImage(images['CAM_BACK_RIGHT'])
