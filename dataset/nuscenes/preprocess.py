#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from dataset.nuscenes.image_class import SingleCameraImage, CameraImages
from dataset.nuscenes.lidar_class import Lidar
from dataset.nuscenes.scene_class import Frame, Scene
import utils.utils as utils


logger = logging.getLogger(__name__)


def get_image_with_token(nusc: NuScenes, token: str) -> SingleCameraImage:
    camera_data = nusc.get("sample_data", token)
    single_camera_image_input_data = {
        "h": camera_data["height"],
        "w": camera_data["width"],
        "file_path": os.path.join(dataroot, camera_data["filename"]),
        "channel": camera_data["channel"],
        "timestamp": camera_data["timestamp"],
        "is_key_frame": camera_data["is_key_frame"]
    }
    return SingleCameraImage(**single_camera_image_input_data)


def get_lidar_with_token(nusc: NuScenes, token: str) -> Lidar:
    lidar_data = nusc.get("sample_data", token)
    file_path = os.path.join(dataroot, lidar_data["filename"])
    channel = lidar_data["channel"]
    return Lidar(file_path=file_path, channel=channel)


def get_frame_data_with_token(nusc: NuScenes, token: str) -> Frame:
    sample_data = nusc.get("sample", token)
    sensor_data_dict = sample_data["data"]
    camera_channel_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
                           "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    lidar_channel = "LIDAR_TOP"
    images_dict = dict()
    for channel in camera_channel_list:
        images_dict[channel] = get_image_with_token(nusc, sensor_data_dict[channel])
    image_data = CameraImages(images_dict)
    lidar_data = get_lidar_with_token(nusc, sensor_data_dict[lidar_channel])
    return Frame(images=image_data, lidar=lidar_data)


def get_scene_data(nusc: NuScenes, scene: Dict) -> Scene:
    curr_sample_token = scene["first_sample_token"]
    last_sample_token = scene["last_sample_token"]
    frames = []
    while curr_sample_token != last_sample_token:
        frame = get_frame_data_with_token(nusc, curr_sample_token)
        frames.append(frame)
        curr_sample_token = nusc.get("sample", curr_sample_token)["next"]
    return Scene(frames=frames, name=scene["name"], description=scene["description"])


def main(version: str):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    output_dir = os.path.join(dataroot, "preprocess")
    os.makedirs(output_dir, exist_ok=True)
    for scene in tqdm(nusc.scene, desc=f"Processing {version}", total=len(nusc.scene)):
        scene_data = get_scene_data(nusc, scene)
        if not scene_data.exist:
            logger.warning(f"Scene: {scene['name']}, does not exist")
            continue
        utils.gdump(scene_data, os.path.join(output_dir, f"{scene['name']}.pkl"))


if __name__ == "__main__":
    import getpass
    user = getpass.getuser()
    dataroot = f"/Users/{user}/datasets/nuscenes"
    utils.format_logger(logger, "./log/nuscenes_preprocess.log")
    # version = "v1.0-trainval"
    version = "v1.0-mini"
    main(version)
