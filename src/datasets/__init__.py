""""""
from .tartanair import TartanAir
from .nyu import NYUDepthV2
from .midair import MidAir
from .hypersim import HyperSim
from .virtual_kitti import VirtualKitti
from .synscapes import Synscapes
from .synthia import Synthia


def get_dataloader(dataset):
    """Gets the dataloader for the specified dataset"""

    datasets = {
        "tartanair": TartanAir,
        "nyu": NYUDepthV2,
        "midair": MidAir,
        "hypersim": HyperSim,
        "virtualkitti": VirtualKitti,
        "synscapes": Synscapes,
        "synthia": Synthia,
    }

    return datasets[dataset.lower()]
