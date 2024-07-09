#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os
from typing import Tuple

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

from dataset.pascal_voc.dataset import PascalVocDataset
from models.unet import UNet
from train.image_segment.loss import get_loss
from utils.utils import weight_init, format_logger, init_seed, convert_onehot_to_mask

logger = logging.getLogger("train_image_segment")
writer = SummaryWriter("./log/tensorboard")

un_norm = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])


def get_dataloader(data_path: str) -> Tuple[DataLoader, DataLoader]:
    dataset = PascalVocDataset(data_path)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_dataset = DataLoader(val_dataset, batch_size=16, shuffle=False)
    return train_loader, val_dataset


def get_validate_loss(model: UNet, val_loader: DataLoader, device: torch.device) -> float:
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for _, (image, label_image) in enumerate(val_loader):
            image, label_image = image.to(device), label_image.to(device)
            output_image = model(image)
            tran_loss = get_loss(output_image, label_image)
            total_loss += tran_loss.item()
    return total_loss / len(val_loader)


def train():
    device = torch.device("mps")
    output_models_dir = "/Users/panchuheng/models/pascal_voc"
    output_image_dir = "/Users/panchuheng/models/pascal_voc/test_image"
    data_path = "/Users/panchuheng/datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    train_loader, val_loader = get_dataloader(data_path)
    data_loader = DataLoader(PascalVocDataset(data_path), batch_size=16, shuffle=False)
    model = UNet().to(device)
    writer.add_graph(model, torch.randn(1, 3, 256, 256).to(device))
    model.apply(weight_init)
    opt = optim.Adam(model.parameters(), lr=1e-5)
    n_epoch = 20
    cnt = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2)
    for epoch in range(n_epoch):
        model.train()
        total_loss = 0
        for i, (image, label_image) in enumerate(train_loader):
            image, label_image = image.to(device), label_image.to(device)
            output_image = model(image)
            train_loss = get_loss(output_image, label_image)
            total_loss += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            if i % 5 == 0:
                logger.info(f"epoch: {epoch}, iter: {i}, train_loss: {train_loss}")
                writer.add_scalar("tran_loss_not_epoch", train_loss, cnt)
                cnt += 1
            if i % 50 == 0:
                i_img = un_norm(image[0])
                l_img = label_image[0]
                l_img = convert_onehot_to_mask(l_img)
                o_img = output_image[0]
                o_img = convert_onehot_to_mask(o_img)
                img = torch.stack([i_img, l_img, o_img], dim=0)
                save_image(img, os.path.join(output_image_dir, f"{epoch}_{i}.png"))
        avg_train_loss = total_loss / len(data_loader)
        avg_val_loss = get_validate_loss(model, val_loader, device)
        logger.info(f"epoch: {epoch}, avg_train_loss: {avg_train_loss}, avg_val_loss: {avg_val_loss}")
        writer.add_scalars("train_loss_epoch", {"train": avg_train_loss, "val": avg_val_loss}, epoch)
        torch.save(model.state_dict(), os.path.join(output_models_dir, f"model_{epoch}.pth"))
        scheduler.step(avg_train_loss)
    writer.close()


if __name__ == "__main__":
    format_logger(logger, "./log/image_segment.log")
    logger.info("start train image segment")
    init_seed(1234)
    train()
