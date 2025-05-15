"Dataloading utilities for Synscapes dataset"

import json
import os

import numpy as np
from PIL import Image
import cv2

from datasets.dataset import Dataset

CLASSES = {
    -1: (0, 0, 0),
    0: (111, 74, 0),
    1: (81, 0, 81),
    2: (128, 64, 128),
    3: (244, 35, 232),
    4: (250, 170, 160),
    5: (230, 150, 140),
    6: (70, 70, 70),
    7: (102, 102, 156),
    8: (190, 153, 153),
    9: (180, 165, 180),
    10: (150, 100, 100),
    11: (150, 120, 90),
    12: (153, 153, 153),
    13: (153, 153, 153),
    14: (250, 170, 30),
    15: (220, 220, 0),
    16: (107, 142, 35),
    17: (152, 251, 152),
    18: (70, 130, 180),
    19: (220, 20, 60),
    20: (255, 0, 0),
    21: (0, 0, 142),
    22: (0, 0, 70),
    23: (0, 60, 100),
    24: (0, 0, 90),
    25: (0, 0, 110),
    26: (0, 80, 100),
    27: (0, 0, 230),
    28: (119, 11, 32),
    29: (0, 0, 142),
}

"Utility Functions_____________________________________________________________"


def gen_paths_from_id(root, idx):
    rgb_path = os.path.join(root, f"img/rgb/{idx}.png")
    depth_path = os.path.join(root, f"img/depth/{idx}.npy")
    seg_path = os.path.join(root, f"img/class/{idx}.png")

    return {"image_l": rgb_path, "depth_l": depth_path, "seg_l": seg_path}


# MidAir Loader _____________________________________________________________


class Synscapes(Dataset):
    "Dataloader for synscapes dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)
        self.normalize_sky = args.get("normalize_sky", False)

    def gen_file_list(self, dataset_path, split_file, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Depth L
            - Image L

        """
        print("Generating Synscapes Filelist")
        with open(split_file, "r") as handler:
            split_ids = json.load(handler)[split]

        return [gen_paths_from_id(dataset_path, id) for id in split_ids]

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"

        if feature.startswith("depth"):
            img = np.load(image_path).astype("float32")
            img = self._crop_center(img)
            if self.normalize_sky:
                img[img == -1] = np.max(img)
            img = Image.fromarray(img.astype(np.float32))

        if feature.startswith("image"):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._crop_center(img)
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
            resample = (
                Image.BICUBIC if feature.startswith("image") else Image.NEAREST
            )
            img = img.resize(resize_shape, resample=resample)
        return img

    def _mask_sky(self, depth):
        return depth # Sky has no annotations