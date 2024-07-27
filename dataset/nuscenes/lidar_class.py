#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np


class Lidar:
    def __init__(self, file_path: str, channel: str):
        self.exist = os.path.exists(file_path)
        self.file_path = file_path
        self.channel = channel
        self.points = None
        if self.exist:
            self.points = self.get_lidar_from_file(file_path)

    def get_lidar_from_file(self, file_path):
        scan = np.fromfile(file_path, dtype=np.float32)
        return scan.reshape((-1, 5))[:, :4]
