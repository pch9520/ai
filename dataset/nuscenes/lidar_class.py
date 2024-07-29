#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from typing import Tuple

import numpy as np


class Lidar:
    def __init__(self, file_path: str, channel: str):
        self.exist = os.path.exists(file_path)
        self.file_path = file_path
        self.channel = channel
        self.points = None
        self.label = None

    def get_lidar_from_file(self) -> Tuple[np.ndarray, np.ndarray]:
        scan = np.fromfile(self.file_path, dtype=np.float32)
        return scan.reshape((-1, 5))[:, :4]
