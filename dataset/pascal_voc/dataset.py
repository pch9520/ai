#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import pad_and_resize_img


transform = transforms.Compose([
    transforms.ToTensor()
])

transform_with_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# TODO: add data augmentation module


class PascalVocDataset(Dataset):
    def __init__(self, path: str) -> None:
        super(PascalVocDataset, self).__init__()
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self) -> int:
        return len(self.name)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = pad_and_resize_img(segment_path)
        image = pad_and_resize_img(image_path, mode='RGB')
        segment_image = transform(segment_image) * 255
        segment_image = torch.squeeze(segment_image.long(), dim=0)
        segment_image = torch.where(segment_image == 255, 0, segment_image)
        segment_image = F.one_hot(segment_image, num_classes=21).permute(2, 0, 1)
        return transform_with_norm(image), segment_image


if __name__ == '__main__':
    root_path = "/Users/panchuheng/datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    dataset = PascalVocDataset(root_path)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
