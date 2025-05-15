"Dataloading utilities for MidAir dataset"

from glob import glob
import json
import os
from pathlib import Path
import cv2

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from datasets.dataset import Dataset

from datasets.image_transforms import ImageTransformer

"Utility Functions_____________________________________________________________"


def get_split_from_json(split, file):
    with open(file, "r") as f:
        return json.load(f)[split]


def is_valid_folder(folder_path):
    """Validates if the folder contains the expected subfolders"""

    # Get subfolders
    subfolders = [f for f in glob(folder_path + "/*") if os.path.isdir(f)]
    subfolders = [f.split("/")[-1] for f in subfolders]

    # Verify if subfolders exists
    expected_subfolders = ["Easy", "Hard"]
    return all([sf in subfolders for sf in expected_subfolders])


def get_scenes_paths(dataset_path):
    "Get the path of all the scene folders found in the dataset"

    # Verify if dataset_path has a / in the end
    dataset_path = (
        dataset_path + "/" if not dataset_path.endswith("/") else dataset_path
    )

    # Get all folder candidate in dataset root
    folders = [x for x in glob(dataset_path + "*") if not x[:-3].endswith(".")]

    # Filter no-folders
    folders = [f for f in folders if os.path.isdir(f)]

    # Validate if is a scene folder
    folders = [f for f in folders if is_valid_folder(f)]

    return folders


def get_path_ids(climate_root, trajec):
    "Get the id of all the available files in a path"

    all_depths = glob(os.path.join(climate_root, "depth", trajec) + "/*.PNG")
    return [Path(x).stem for x in all_depths]


def get_data_from_id(id, climate_root, trajec):
    "Get the path from all the data from a id"

    return {
        "depth_l": os.path.join(climate_root, "depth", trajec, f"{id}.PNG"),
        "image_l": os.path.join(climate_root, "color_left", trajec, f"{id}.JPEG"),
        "image_r": os.path.join(climate_root, "color_right", trajec, f"{id}.JPEG"),
        "seg_l": os.path.join(climate_root, "segmentation", trajec, f"{id}.PNG"),
    }


# MidAir Loader _____________________________________________________________


class MidAir(Dataset):
    "Dataloader for tartanair dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)
        self.scenes = get_split_from_json(split, split_json)
        self.normalize_sky = args.get("normalize_sky", False)

    def gen_file_list(self, dataset_path, split_json, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Scene name
            - Depth L
            - Image L
            - Image R
            - Segmentation L

        Scenes is a dict:
            map{
                climate[
                    trajectories
                ]
            }
        """
        scenes = get_split_from_json(split, split_json)
        file_list = []
        for map in scenes:
            for climate in scenes[map]:
                climate_root = os.path.join(dataset_path, map, climate)
                for trajec in scenes[map][climate]:
                    ids = get_path_ids(climate_root, trajec)

                    for id in ids:
                        all_paths_id = get_data_from_id(id, climate_root, trajec)
                        file_list.append(all_paths_id)
        return file_list

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"
        img = Image.open(image_path)

        if feature.startswith("depth"):
            img = np.asarray(img, np.uint16)
            img = img.astype(np.float16)
            # img = 255 / img
            if self.mask_sky:
                self._mask_sky(img)
            if self.normalize_sky:
                img[img == 31740] = np.max(img[img != 31740])

            img = Image.fromarray(img.astype(np.float32))
        if feature.startswith("seg"):
            img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            img = Image.fromarray(img)

        # Resizes if shape is provided
        img = np.array(img)
        img = self._crop_center(img)
        img = Image.fromarray(img)
        if resize_shape:
            img = img.resize(resize_shape, resample=Image.BICUBIC)

        return img

    def _mask_sky(self, depth):
        depth[depth == 31740] = -1
        return depth
