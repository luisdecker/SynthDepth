"Dataloading utilities for KITTI dataset"

from glob import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader
from datasets.dataset import Dataset

from datasets.image_transforms import ImageTransformer

"Utility Functions_____________________________________________________________"


def load_filelist(filepath):
    "Loads a list of files from a text file"

    with open(filepath, "r") as f:
        data = f.readlines()
    data = [line.split(" ") for line in data]
    return data


# MidAir Loader _____________________________________________________________


class Kitti(Dataset):
    "Dataloader for kitti dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)

    def gen_file_list(self, dataset_path, split_file, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Depth L
            - Image L

        """
        file_list = []
        # Get data from split filelist
        raw_file_list = load_filelist(split_file)
        for sample in raw_file_list:
            image, depth = sample
            if depth.endswith("\n"):
                depth = depth[:-1]
            image = os.path.join(dataset_path, image)
            depth = os.path.join(dataset_path, depth)
            file_list.append({"image_l": image, "depth_l": depth})

        return file_list

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"

        if feature.startswith("depth"):
            img = np.array(Image.open(image_path), dtype=int)
            # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype("float32")
            img = self._crop_center(img)
            img[img < 0] = 0

            img = Image.fromarray(img.astype(np.float32) / 256)

        if feature.startswith("image"):
            img = cv2.imread(image_path)
            img = self._crop_center(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        # Resizes if shape is provided
        if resize_shape and not self.resize_to_height:
            img = img.resize(
                resize_shape,
                resample=(
                    Image.BICUBIC if feature.startswith("image") else Image.NEAREST
                ),
            )
        if self.resize_to_height:
            img = Image.fromarray(self._resize_to_height(np.array(img)))

        return img
