"Baseclass for datasets"

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2

from datasets.image_transforms import ImageTransformer


class Dataset:
    "Baseclass for datasets"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        self.dataset_root = dataset_root
        self.file_list = self.gen_file_list(self.dataset_root, split_json, split)
        self.target_size = args.get("target_size")
        self.features = args["features"]
        self.split = split
        self.depth_clip = args.get("depth_clip")
        self.mask_sky = args.get("mask_sky")
        self.augmentation = args.get("augmentation", False)
        self.crop_center = args.get("crop_center", False)
        self.resize_to_height = args.get("resize_to_height", False)
        self.normalize_sky = args.get("normalize_sky", False)

        if self.mask_sky and self.normalize_sky:
            print("[Warning!] Dataset loaded with both normalize_sky and mask_sky!")

        assert not (
            self.resize_to_height and self.crop_center
        ), "Cannot crop center and resize to height"

        self.image_transformer = ImageTransformer(
            self.split, augmentation=self.augmentation
        ).get_transform()

    def __len__(self):
        "Get number of samples of dataset"
        return len(self.file_list)

    def __getitem__(self, idx):
        "Gets one dataset sample"
        data_paths = self.file_list[idx]

        input_features, label_features = self.features
        feats = set().union(input_features).union(label_features)

        data = self.get_data_from_features(data_paths, feats)

        input_data = torch.stack([data[f] for f in input_features])

        label_data = torch.stack([data[f] for f in label_features])

        return input_data, label_data

    def _crop_center(self, img):
        "Generates a central crop of the image"
        if not self.crop_center:
            return img
        img = np.array(img)
        height, width = img.shape[:2]
        crop1 = (width - height) // 2
        crop2 = (width - height) // 2 + height
        return img[:, crop1:crop2]

    def _resize_to_height(self, img):
        img_height, img_width = img.shape[:2]

        target_height = self.target_size[1]
        # Resize the image to match the target height while keeping proportions
        scale_ratio = target_height / img_height  
        new_width = int(img_width * scale_ratio)
        img = cv2.resize(np.array(img), (new_width, target_height))

        # Crop the width to the previous multiple of 32
        target_width = (new_width // 32) * 32
        if target_width <= 0:
            raise ValueError("Image width is too small to be cropped to a multiple of 32.")

        img = img[:, :target_width]  # Crop the image width
        return img

    def get_data_from_features(self, data_paths, features):
        data = {}
        for feature in features:
            read_data = self._load_image(data_paths[feature], self.target_size, feature)

            # Depth clipping
            if feature.startswith("depth") and self.depth_clip:
                depth = np.array(read_data)
                np.clip(depth, a_min=0, a_max=self.depth_clip)
                read_data = Image.fromarray(depth).convert("F")

            data[feature] = read_data

        return self.image_transformer(data)

    def build_dataloader(self, shuffle, batch_size, num_workers):
        "Builds a torch dataloader"
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            # prefetch_factor=2,
        )
