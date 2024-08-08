#!/usr/bin/python
# -*- coding: utf-8 -*-

# copy from https://github.com/itberrios/3D/blob/main/point_net/shapenet_dataset.py

from collections import defaultdict
import json
import os
from typing import List

import numpy as np
import open3d as o3
from PIL import Image
import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):

    def __init__(self, root: str,
                 split: str,
                 npoints: int = 2500,
                 classification: bool = False,
                 class_choice: List[str] = [],
                 image=None,
                 normalize=True) -> None:
        super(ShapeDataset, self).__init__()
        self.root = root
        self.split = split.lower()
        self.npoints = npoints
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.image = image
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if class_choice:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        self.meta = defaultdict(list)
        for item in self.cat:
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_seg_img = os.path.join(self.root, self.cat[item], 'seg_img')

            if self.split == 'train':
                split_file = os.path.join(self.root, "train_test_split", 'shuffled_train_file_list.json')
            elif self.split == 'test':
                split_file = os.path.join(self.root, "train_test_split", 'shuffled_test_file_list.json')
            elif self.split == 'val':
                split_file = os.path.join(self.root, "train_test_split", 'shuffled_val_file_list.json')
            else:
                raise ValueError("Invalid split name: {}".format(self.split))

            with open(split_file, 'r') as f:
                split_data = json.load(f)

            pts_names = [token.split("/")[-1] + ".pts" for token in split_data if self.cat[item] in token]
            for fn in pts_names:
                token = os.path.basename(fn).split('.')[0]
                self.meta[item].append((os.path.join(dir_point, token + ".pts"),
                                        os.path.join(dir_seg, token + '.seg'),
                                        os.path.join(dir_seg_img, token + '.png')))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

        if not self.classification:
            self.num_seg_classes = self.get_seg_classes_num()

    def get_seg_classes_num(self) -> int:
        class_set = set()
        for i in range(len(self.datapath)):
            _, _, seg_file, _ = self.datapath[i]
            class_set |= set(np.unique(np.loadtxt(seg_file).astype(np.uint8)))
        return len(class_set)

    def __get_item__(self, index):
        item, point_path, seg_path, seg_img_path = self.datapath[index]
        cls_ = self.classes[item]
        points = np.asarray(o3.io.read_point_cloud(point_path, format='xyz').points, dtype=np.float32)
        seg = np.loadtxt(seg_path).astype(np.uint8)
        image = Image.open(seg_img_path)

        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        points = points[choice, :]
        seg = seg[choice]
        points = torch.from_numpy(points)
        seg = torch.from_numpy(seg)
        cls_ = torch.from_numpy(np.array([cls_]).astype(np.uint8))

        if self.split != 'test':
            points += torch.randn(points.shape) / 100
            points = self.random_rotate(points)

        if self.normalize:
            points = self.normalize_points(points)

        if self.classification:
            if self.image:
                return points, cls_, image
            else:
                return points, cls_
        else:
            if self.image:
                return points, seg, image
            else:
                return points, seg

    @staticmethod
    def random_rotate(points: torch.Tensor):
        theta = torch.FloatTensor(1).uniform_(-np.pi, np.pi)
        rotation_matrix = torch.Tensor([torch.cos(theta), 0, torch.sin(theta),
                                        [0, 1, 0],
                                        [-torch.sin(theta), 0, torch.cos(theta)]])
        return torch.matmul(points, rotation_matrix)

    @staticmethod
    def normalize_points(points: torch.Tensor):
        points = points - points.mean(axis=0)[0]
        points /= (points.max(axis=0)[0] - points.min(axis=0)[0])
        return points
