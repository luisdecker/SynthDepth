"Dataloading utilities for Hypersim dataset"

import json
import os

import skimage
import numpy as np
from PIL import Image
import h5py

from datasets.dataset import Dataset


"Utility Functions_____________________________________________________________"


def get_split_from_json(split, file):
    with open(file, "r") as f:
        return json.load(f)[split]


def get_data_from_row(dataset_path, scene, camera, id):
    """Gets the paths for all data for a given id"""

    basepath = os.path.join(dataset_path, scene, "images")
    rgb = os.path.join(
        basepath,
        f"scene_{camera}_final_preview",
        f"frame.{id:0>4}.tonemap.jpg",
    )
    depth = os.path.join(
        basepath,
        f"scene_{camera}_geometry_hdf5",
        f"frame.{id:0>4}.depth_meters.hdf5",
    )  # size? unit?
    semantic = os.path.join(
        basepath,
        f"scene_{camera}_geometry_hdf5",
        f"frame.{id:0>4}.semantic.hdf5",
    )  # size?

    return {"depth_l": depth, "image_l": rgb, "seg_l": semantic}


"MidAir Loader _____________________________________________________________"


class HyperSim(Dataset):
    "Dataloader for hypersim dataset"

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
            - Image L
            - Image R
            - Segmentation L


        """
        scenes = get_split_from_json(split, split_json)
        file_list = [get_data_from_row(dataset_path, *row) for row in scenes]

        return file_list

    def _load_image(self, image_path, resize_shape, feature):
        """Load a sample"""
        if image_path.endswith(".jpg"):
            # Loads image
            assert os.path.isfile(image_path), f"{image_path} is not a file!"
            img = Image.open(image_path)

            if self.crop_center:
                img = Image.fromarray(self._crop_center(img))

            # Resizes if shape is provided
            if resize_shape:
                img = img.resize(resize_shape, resample=Image.BICUBIC)

            return img
        else:
            return self._load_hdf5(image_path, resize_shape, feature)

    def _load_hdf5(self, image_path, resize_shape, feature):
        "Load an hdf5 image"
        is_depth = feature.startswith("depth")
        with h5py.File(image_path, locking=False) as f:
            img = f["dataset"][()].astype(np.float32)
            if self.crop_center:
                img = self._crop_center(img)
            if is_depth:
                img = HyperSim._distance_to_depth(img, self.crop_center)

                if np.isnan(img).any():
                    mask = np.isnan(img)
                    try:
                        img = skimage.restoration.inpaint_biharmonic(img, mask)
                    except:
                        print("Deu pau")

            if feature.startswith("image"):  # This clip should be a tonemap
                # clip image
                img = img.clip(max=1)
                img = img * 255
                img = img.astype("uint8")
            img = Image.fromarray(img)

            if resize_shape:
                resample = (
                    Image.BICUBIC if feature.startswith("image") else Image.NEAREST
                )
                img = img.resize(resize_shape, resample=resample)

            return img

    @staticmethod
    def _distance_to_depth(distance, crop_center=False):
        "Converts from distance to camera point to distance to camera plane"

        intWidth = 1024
        intHeight = 768
        if crop_center:
            intWidth = 768
        fltFocal = 886.81

        imgPlaneX = (
            np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
            .reshape(1, intWidth)
            .repeat(intHeight, 0)
            .astype(np.float32)[:, :, None]
        )

        imgPlaneY = (
            np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
            .reshape(intHeight, 1)
            .repeat(intWidth, 1)
            .astype(np.float32)[:, :, None]
        )

        imgPlaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)

        imagePlane = np.concatenate([imgPlaneX, imgPlaneY, imgPlaneZ], 2)

        return distance / np.linalg.norm(imagePlane, 2, 2) * fltFocal
