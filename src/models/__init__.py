from models.losses import FocalLoss, CombinedLoss, MidasLoss, MidasLossMedian
from .simple_encoder import SimpleEncoder
from .convNext import ConvNext

import torch.nn as nn
from torchgeometry.losses import SSIM


def get_loss(loss_name, **kwargs):
    """gets a loss object

    Args:
        loss_name (string): Name of the loss
    kwargs:
        optional arguments for the losses
    """
    if loss_name is None:
        return None

    losses = {
        "huber": nn.HuberLoss,
        "ssim": SSIM,
        "combined": CombinedLoss,
        "focal": FocalLoss,
        "midas": MidasLoss,
        "midas-median": MidasLossMedian,
        "mse": nn.MSELoss,
    }
    loss = losses[loss_name.lower()]
    loss = loss(**kwargs)
    return loss


# TODO def get_model()
