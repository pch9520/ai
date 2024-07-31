#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from typing import Tuple

import numpy as np


class Lidar:
    def __init__(self, file_path: str, channel: str, lidar_seg_path: str = None, panoptic_path: str = None):
        self.exist = os.path.exists(file_path)
        self.file_path = file_path
        self.channel = channel
        self.lidar_seg_path = lidar_seg_path
        self.panoptic_path = panoptic_path

    def get_lidar_from_file(self) -> Tuple[np.ndarray, np.ndarray]:
        assert os.path.exists(self.file_path), f"Lidar file: {self.file_path}, not exist"
        scan = np.fromfile(self.file_path, dtype=np.float32)
        return scan.reshape((-1, 5))[:, :4]

    def get_lidar_seg_label_from_file(self) -> np.ndarray:
        assert self.lidar_seg_path is not None and os.path.exists(self.lidar_seg_path), \
            f"Label file: {self.lidar_seg_path}, not exist"
        return np.fromfile(self.lidar_seg_path, dtype=np.uint8)

    def get_panoptic_label_from_file(self) -> np.ndarray:
        assert self.panoptic_path is not None and os.path.exists(self.panoptic_path), \
            f"Label file: {self.panoptic_path}, not exist"
        return np.load(self.panoptic_path)["data"] // 1_000
