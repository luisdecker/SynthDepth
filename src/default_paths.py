"Default split files"

DEFAULT_SPLIT_FILES = {
    "hypersim": "splits/hypersim_splits_nonan_noblack.json",
    "kitti": "splits/kitti_eigen_val.txt",
    "nyu": "splits/nyu.json",
    "midair": "splits/midair_splits.json",
    "synscapes": "splits/synscapes_split.json",
    "synthia": "splits/synthia_splits.json",
    "tartanair": "splits/tartanair_scenes_splits.json",
    "virtualkitti": "splits/virtual_kitty_splits.json",
}

# !!! CONFIGURE THIS !!!
# Configure the following paths with the root folder of the datasets.
# The folders must be a folder in your machine, or mounted as one. 

DEFAULT_DATASET_ROOT = {
    "hypersim": "/hadatasets/hypersim",
    "kitti": "/hadatasets/kitti",
    "nyu": "/hadatasets/nyu",
    "midair": "/hadatasets/midair/MidAir",
    "synscapes": "/hadatasets/synscapes",
    "synthia": "/hadatasets/SYNTHIA-AL/",
    "tartanair": "/hadatasets/tartan-air/tartan-air/train",
    "virtualkitti": "/hadatasets/virtual_kitty",
}
