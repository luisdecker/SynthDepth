"Dataloading utilities for virtual KITTI 2 dataset"

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from datasets.dataset import Dataset


"Utility Functions_____________________________________________________________"

CLASSES = {
    -1: [0, 0, 0],
    0: [210, 0, 200],
    1: [90, 200, 255],
    2: [0, 199, 0],
    3: [90, 240, 0],
    4: [140, 140, 140],
    5: [100, 60, 100],
    6: [250, 100, 255],
    7: [255, 255, 0],
    8: [200, 200, 0],
    9: [255, 130, 0],
    10: [80, 80, 80],
    11: [160, 60, 60],
    12: [255, 127, 80],
    13: [0, 139, 139],
}


def load_filelist(filepath):
    "Loads a list of files from a text file"

    with open(filepath, "r") as f:
        data = f.readlines()
    data = [line.split(" ") for line in data]
    return data


def get_image_numbers(filelist):
    return [Path(file).stem.split("_")[-1] for file in filelist]


def get_rgb_and_depth(scenes, dataset_path):
    variations = [
        "15-deg-left",
        "15-deg-right",
        "30-deg-left",
        "30-deg-right",
        "clone",
        "fog",
        "morning",
        "overcast",
        "rain",
        "sunset",
    ]

    all_files = []
    for scene in scenes:
        for variation in variations:
            basepath_rgb = os.path.join(
                dataset_path, f"Scene{scene}/{variation}/frames/rgb/Camera_0/"
            )
            basepath_depth = os.path.join(
                dataset_path,
                f"Scene{scene}/{variation}/frames/depth/Camera_0/",
            )

            basepath_seg = os.path.join(
                dataset_path,
                f"Scene{scene}/{variation}/frames/classSegmentation/Camera_0/",
            )

            rgb_folder_files = [
                f"{file}" for file in list(Path(basepath_rgb).glob("*.jpg"))
            ]

            depth_folder_files = [
                os.path.join(basepath_depth, f"depth_{n}.png")
                for n in get_image_numbers(rgb_folder_files)
            ]

            seg_folder_files = [
                os.path.join(basepath_seg, f"classgt_{n}.png")
                for n in get_image_numbers(rgb_folder_files)
            ]
            all_files.extend(
                zip(rgb_folder_files, depth_folder_files, seg_folder_files)
            )
    return all_files


# MidAir Loader _____________________________________________________________


class VirtualKitti(Dataset):
    "Dataloader for virtual kitti dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)
        self.num_classes = 15
        self.normalize_sky = args.get("normalize_sky", False)

    def gen_file_list(self, dataset_path, split_file, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Depth L
            - Image L

        """
        print("Generating Virtual Kitti Filelist")
        with open(split_file, "r") as handler:
            split_scenes = json.load(handler)[split]

        all_files = get_rgb_and_depth(split_scenes, dataset_path)

        return [{"depth_l": d, "image_l": i, "seg_l": s} for i, d, s in all_files]

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"

        if feature.startswith("depth"):
            img = cv2.imread(
                image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            ).astype("float32")
            img[img > 0] /= 100  # cm to m
            if self.mask_sky:
                img = self._mask_sky(img)
            # img[img >= 65535] = -1  # sky not valid

            if self.normalize_sky:
                img[img == 655.3500] = np.max(img[img != 655.3500])
                
            img = self._crop_center(img) if self.crop_center else img

            img = Image.fromarray(img.astype(np.float32))

        if feature.startswith("image"):
            img = cv2.imread(image_path)
            img = self._crop_center(img) if self.crop_center else img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        if feature.startswith("seg"):
            img = cv2.imread(image_path)
            new_img = np.zeros(img.shape[:-1])
            for cls, pixel in CLASSES.items():
                new_img[np.all(img == pixel, axis=-1)] = cls
            new_img = self._crop_center(new_img)
            img = Image.fromarray(new_img)

        # Resizes if shape is provided
        if resize_shape:
            resample = Image.BICUBIC if feature.startswith("image") else Image.NEAREST
            img = img.resize(resize_shape, resample=resample)

        return img

    def _mask_sky(self, depth):
        depth[depth == 655.3500] = -1
        return depth
