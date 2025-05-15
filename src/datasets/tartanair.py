"Dataloading utilities for TartanAir dataset"

from glob import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from datasets.dataset import Dataset
import cv2


from datasets.image_transforms import ImageTransformer

SKY_INDEXES = {
    "neighborhood": 146,
    "abandonedfactory": 196,
    "abandonedfactory_night": 196,
    "amusement": 182,
    "carwelding": None,
    "endofworld": 146,
    "gascola": 112,
    "hospital": 200,
    "japanesealley": 130,
    "ocean": 231,
    "office": 146,
    "office2": 146,
    "oldtown": 223,
    "seasidetown": 130,
    "seasonsforest": 196,
    "seasonsforest_winter": 146,
    "soulcity": 130,
    "westerndesert": 196,
}

# fmt: off
"""DATASET STRUCTURE
-scene1
    - Easy
        - Pxxx
            - depth_left
                - kkkk_left_depth.npy
                - kkkl_left_depth.npy
                - ...
            - depth_right
                - kkkk_right_depth.npy
                - kkkl_right_depth.npy
                - ...
            - image_left
                - kkkk_left.png
                - kkkl_left.png
            - image_right
                - kkkk_right.png
                - kkkl_right.png
            - seg_left
                - kkkk_left_seg.npy
                - kkkl_left_seg.npy
            - seg_right
                - kkkk_right_seg.npy
                - kkkl_right_seg.npy
        - Pxxy
    - Hard
-scene2 
...


"""


# fmt: on
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


def get_path_ids(path):
    "Get the id of all the available files in a path"
    all_depths = glob(os.path.join(path, "depth_left") + "/*")
    return [x.split("/")[-1].split("_")[0] for x in all_depths]


def get_data_from_id(id, path):
    "Get the path from all the data from a id"

    return {
        "depth_l": os.path.join(path, "depth_left", f"{id}_left_depth.npy"),
        "depth_r": os.path.join(path, "depth_right", f"{id}_right_depth.npy"),
        "image_l": os.path.join(path, "image_left", f"{id}_left.png"),
        "image_r": os.path.join(path, "image_right", f"{id}_right.png"),
        "seg_l": os.path.join(path, "seg_left", f"{id}_left_seg.npy"),
        "seg_r": os.path.join(path, "seg_right", f"{id}_right_seg.npy"),
    }


# Tartanair Loader _____________________________________________________________


class TartanAir(Dataset):
    "Dataloader for tartanair dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)
        self.scenes = get_split_from_json(split, split_json)

    def gen_file_list(self, dataset_path, split_json, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Scene name
            - Depth L
            - Depth R
            - Image L
            - Image R
            - Segmentation L
            - Segmentation R
        """
        scenes = get_split_from_json(split, split_json)
        file_list = []
        scene_folders = get_scenes_paths(dataset_path)
        found_scenes = [scene.split("/")[-1] for scene in scene_folders]
        assert all(
            scene in found_scenes for scene in scenes
        ), f"{scene} not found!\n{found_scenes}"

        for scene in scene_folders:
            scene_name = scene.split("/")[-1]
            if scene_name not in scenes:
                continue

            for difficulty in ["Easy", "Hard"]:
                # Get all the available paths
                paths_root = os.path.join(scene, difficulty)
                paths = glob(paths_root + "/*")

                for path in paths:
                    # Get available image ids
                    ids = get_path_ids(path)
                    for id in ids:
                        # get all the data for this id
                        all_paths_id = get_data_from_id(id, path)

                        # add extra info
                        all_paths_id["scene"] = scene_name

                        file_list.append(all_paths_id)
        return file_list

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        if image_path.endswith(".png"):
            # Loads image
            assert os.path.isfile(image_path), f"{image_path} is not a file!"
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            img = TartanAir._load_npz(image_path)
            if feature.startswith("depth"):
                if self.mask_sky:
                    img = self._mask_sky(img, image_path)
                if self.normalize_sky:
                    img = self._normalize_sky(img, image_path)

                # Remove depth from dark areas
                rgb_sum = cv2.imread(
                    TartanAir._get_rgb_from_depth(image_path),
                ).sum(axis=2)

                img[rgb_sum == 0] = -1

        # Resizes if shape is provided
        img = self._crop_center(img)
        img = Image.fromarray(img)
        if resize_shape:
            resample = Image.BICUBIC if feature.startswith("image") else Image.NEAREST
            img = img.resize(resize_shape, resample=resample)

        return img

    def _normalize_sky(self, depth, image_path):
        """
        Normalize the depth values for sky regions in the depth map.

        This function adjusts the depth values for sky regions in the provided depth map
        based on the segmentation image and scene information derived from the image path.
        Sky regions are identified using predefined indexes for different scenes.

        Args:
            depth (np.ndarray): The depth map to be normalized.
            image_path (str): The file path to the image corresponding to the depth map.

        Returns:
            np.ndarray: The normalized depth map with adjusted sky region values.
        """
        seg_img = TartanAir._get_seg_from_depth(image_path)
        scene = TartanAir._get_scene_from_path(image_path)
        if SKY_INDEXES[scene]:
            seg_img = TartanAir._load_npz(seg_img)
            depth[seg_img == SKY_INDEXES[scene]] = np.max(
                depth[seg_img != SKY_INDEXES[scene]]
            )
        return depth

    def _mask_sky(self, depth, image_path):
        """
        Masks the sky regions in the depth map based on the segmentation image.

        Args:
            depth (np.ndarray): The depth map to be modified.
            image_path (str): The path to the image file used to obtain segmentation and scene information.

        Returns:
            np.ndarray: The modified depth map with sky regions masked (set to -1).
        """

        seg_img = TartanAir._get_seg_from_depth(image_path)
        scene = TartanAir._get_scene_from_path(image_path)
        if SKY_INDEXES[scene]:
            seg_img = TartanAir._load_npz(seg_img)
            depth[seg_img == SKY_INDEXES[scene]] = -1
        return depth

    @staticmethod
    def _load_npz(filepath):
        assert os.path.isfile(filepath), f"{filepath} is not a file!"
        data = np.load(filepath)

        return data

    @staticmethod
    def _get_seg_from_depth(depth_path):
        """Gets a segmentation annotation from the path of a given depth
        annotation"""

        path_list = Path(depth_path).parts[1:]
        # Gets frame number
        frame_number = path_list[-1].split("_")[0]
        scene_path_dir = os.path.join("/", *path_list[:-2])
        seg_dir = os.path.join(scene_path_dir, "seg_left")
        seg_file = os.path.join(seg_dir, f"{frame_number}_left_seg.npy")
        return seg_file

    @staticmethod
    def _get_scene_from_path(image_path):
        """Gets the scene from the path of a image"""
        return Path(image_path).parts[-5]

    @staticmethod
    def _get_rgb_from_depth(depth_path):
        """Gets a rgb image corresponding to a depth map path"""
        path_list = Path(depth_path).parts[1:]
        frame_number = path_list[-1].split("_")[0]
        scene_path_dir = os.path.join("/", *path_list[:-2])
        img_dir = os.path.join(scene_path_dir, "image_left")
        img_file = os.path.join(img_dir, f"{frame_number}_left.png")
        return img_file
