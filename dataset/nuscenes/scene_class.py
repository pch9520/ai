#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import List

from dataset.nuscenes.image_class import CameraImages
from dataset.nuscenes.lidar_class import Lidar


class Frame:
    def __init__(self, images: CameraImages, lidar: Lidar):
        self.images = images
        self.lidar = lidar


class Scene:
    def __init__(self, frames: List[Frame], name: str, description: str) -> None:
        self.frames = frames
        self.name = name
        self.description = description
        self.exist = self.check_is_valid_scene()

    def check_is_valid_scene(self) -> bool:
        exist = True
        camera_images = ["front_image", "front_left_image", "front_right_image", "back_image",
                         "back_left_image", "back_right_image"]
        for frame in self.frames:
            if not frame.lidar.exist:
                return False
            images = frame.images
            for channel in camera_images:
                image = getattr(images, channel)
                if not image.exist:
                    return False
        return exist
