#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

from utils.utils import pad_and_resize_img


transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomRotation(degrees=180),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5)
])

transform_with_norm = v2.Compose([
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# TODO: add data augmentation module


class PascalVocDataset(Dataset):
    def __init__(self, path: str, augment_num: int = 10) -> None:
        super(PascalVocDataset, self).__init__()
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        self.agument_num = augment_num

    def __len__(self) -> int:
        return len(self.name) * self.agument_num

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        initial_seed = torch.initial_seed()
        index %= len(self.name)
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = pad_and_resize_img(segment_path)
        image = pad_and_resize_img(image_path, mode='RGB')
        torch.manual_seed(index)
        image, segment_image = transform(image, segment_image)
        torch.manual_seed(initial_seed)
        image = transform_with_norm(image)
        segment_image = transform(segment_image) * 255
        segment_image = torch.squeeze(segment_image.long(), dim=0)
        segment_image = torch.where(segment_image == 255, 0, segment_image)
        segment_image = F.one_hot(segment_image, num_classes=21).permute(2, 0, 1)
        return image, segment_image


if __name__ == '__main__':
    root_path = "/Users/panchuheng/datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    dataset = PascalVocDataset(root_path)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
