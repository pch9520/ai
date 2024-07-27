#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import List

from dataset.nuscenes.image_class import CameraImages


class Frame:
    def __init__(self, images: CameraImages, lidar):
        self.images = images
        self.lidar = lidar


class Scene:
    def __init__(self, frames: List[Frame]):
        self.frames = frames
