#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

import torch
from torchvision.utils import save_image

from dataset.pascal_voc.dataset import get_dataloader
from models.unet import UNet
import utils.utils as utils


def test(model_path: str, input_path: str) -> None:
    output_path = os.path.join(input_path, "output")
    os.makedirs(output_path, exist_ok=True)
    device = utils.get_torch_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    data_loader = get_dataloader(input_path, is_train=False)
    model.eval()
    for _, (image, file_name) in enumerate(data_loader):
        print(image.shape)
        print(file_name)
        output_image = model(image.to(device))
        for index, o_img in enumerate(output_image):
            save_image(
                utils.convert_onehot_to_mask(o_img), os.path.join(output_path, file_name[index]))


if __name__ == "__main__":
    assert len(sys.argv) > 2, "sys.path must contain at least 3 elements"
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    test(model_path, input_path)
