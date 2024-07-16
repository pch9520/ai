#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

from utils.utils import pad_and_resize_img


def get_dataloader(data_path: str, is_train: bool = True) -> Tuple[DataLoader, Union[DataLoader, str]]:
    if is_train:
        augment_num = 10
    else:
        augment_num = 1
    dataset = PascalVocDataset(data_path, augment_num, is_train)
    if is_train:
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        val_dataset = DataLoader(val_dataset, batch_size=16, shuffle=False)
        return train_loader, val_dataset
    return DataLoader(dataset, batch_size=16, shuffle=False)


train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomRotation(degrees=180),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5)
])

transform_with_norm = v2.Compose([
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PascalVocDataset(Dataset):
    def __init__(self, path: str, augment_num: int = 10, is_train: bool = True) -> None:
        super(PascalVocDataset, self).__init__()
        self.path = path
        if is_train:
            self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        else:
            self.name = os.listdir(path)
            self.name = [name for name in self.name if name.endswith('.jpg') or name.endswith(".png")]
        self.augment_num = augment_num
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.name) * self.augment_num

    def __getitem__(self, index) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        if not self.is_train:
            file_name = self.name[index]
            img_path = os.path.join(self.path, file_name)
            return test_transform(pad_and_resize_img(img_path, mode='RGB')), file_name
        index %= len(self.name)
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = pad_and_resize_img(segment_path)
        image = pad_and_resize_img(image_path, mode='RGB')
        initial_seed = torch.initial_seed()
        torch.manual_seed(index)
        image, segment_image = train_transform(image, segment_image)
        torch.manual_seed(initial_seed)
        image = transform_with_norm(image)
        segment_image = train_transform(segment_image) * 255
        segment_image = torch.squeeze(segment_image.long(), dim=0)
        segment_image = torch.where(segment_image == 255, 0, segment_image)
        segment_image = F.one_hot(segment_image, num_classes=21).permute(2, 0, 1)
        return image, segment_image


if __name__ == '__main__':
    import getpass
    user = getpass.getuser()
    root_path = f"/Users/{user}/datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    dataset = PascalVocDataset(root_path)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
