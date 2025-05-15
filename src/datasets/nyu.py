import os
import json

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from datasets.image_transforms import ImageTransformer


def _crop_center(img):
    "Generates a central crop of the image"
    height, width = img.shape[:2]
    crop1 = (width - height) // 2
    crop2 = (width - height) // 2 + height
    return img[:, crop1:crop2]


class NYUDepthV2:
    "Dataloader for nyu dataset"

    def __init__(self, dataset_root, split, split_json, **kwargs):
        """"""

        self.dataset_root = dataset_root
        self.split = split
        self.split_json = split_json
        self.split_index = self.load_splits()

        self.target_size = kwargs.get("target_size")
        self.features = kwargs["features"]
        self.depth_clip = kwargs.get("depth_clip")
        self.mask_sky = kwargs.get("mask_sky")
        self.crop_center = kwargs.get("crop_center", False)
        self.augmentation = kwargs.get("augmentation", False)

        self.image_transformer = ImageTransformer(
            self.split, augmentation=self.augmentation
        ).get_transform()

        self.images, self.depths = self.load_data()

    def load_splits(self):
        "Load split info"
        with open(self.split_json, "r") as handler:
            splits = json.load(handler)
        return [int(index) - 1 for index in splits[self.split]]

    def load_data(self):
        "Loads the dataset into ram. It is small, should fit"
        dataset = h5py.File(
            os.path.join(self.dataset_root, "nyu_depth_v2_labeled.mat"), locking=False
        )

        images = dataset.get("images")
        images = images[:, :, 7:-7, 7:-7]
        images = np.transpose(images, axes=(0, 3, 2, 1))  # n, 3, h, w
        images_resized = []
        for image in images:
            if self.crop_center:
                image = _crop_center(image)
            image = Image.fromarray(image)
            images_resized.append(
                np.array(
                    image.resize(self.target_size, resample=Image.BICUBIC)
                    if self.target_size
                    else image
                )
            )
        images = np.array(images_resized)

        depths = dataset.get("depths")
        depths = depths[:, 7:-7, 7:-7]
        depths = np.transpose(depths, axes=(0, 2, 1))
        depths_resized = []
        for depth in depths:
            if self.crop_center:
                depth = _crop_center(depth)
            depth = Image.fromarray(depth).convert("F")
            depths_resized.append(
                np.array(depth.resize(self.target_size, Image.BICUBIC))
                if self.target_size
                else np.array(depth)
            )
        depths = np.array(depths_resized)

        return images, depths

    def __len__(self):
        return len(self.split_index)

    def __getitem__(self, idx):
        "Gets one sample"
        index = self.split_index[idx]
        input_features, label_features = self.features

        input_data = {}
        for feature in input_features:
            if feature != "image_l":
                print(f"NYU has no {feature}")
                raise NotImplementedError
            loaded_data = self.images[index]
            input_data[feature] = loaded_data
        input_data = self.image_transformer(input_data)
        input_data = torch.stack([input_data[f] for f in input_features])

        label_data = {}
        for feature in label_features:
            if feature != "depth_l":
                print(f"NYU has no {feature}")
                raise NotImplementedError
            label_data[feature] = self.depths[index]
        label_data = self.image_transformer(label_data)
        label_data = torch.stack([label_data[f] for f in label_data])

        return input_data, label_data

    def build_dataloader(self, shuffle, batch_size, num_workers):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
