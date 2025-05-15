"""Depth estimation metrics"""

from pydoc import cli
import torch
from torchmetrics import Metric, MeanSquaredError, MeanSquaredLogError
from functools import partial

from models.losses import MidasLossMedian, compute_scale_and_shift


def get_metric(metric):
    """Gets a metric class by a string identifier"""

    return {
        "a1": Alpha1,
        "a2": Alpha2,
        "a3": Alpha3,
        "logrmse": MeanSquaredLogError,
        "absrel": AbsoluteRelative,
        "squaredrel": AbsoluteRelativeSquared,
        "rmse": RMSE,
    }[metric.lower()]


class AlphaError(Metric):
    """Computes the alpha error metric in a given power, considering 1.25 as
    threshold"""

    def __init__(self, power):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.power = power

    def update(self, pred, gt):
        "Updates the internal states"
        pred = pred[gt > 0]
        gt = gt[gt > 0]
        thresh = torch.max((gt / pred), (pred / gt))
        sum = (thresh < (1.25**self.power)).float().sum()
        self.sum += sum
        self.n += pred.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


class Alpha1(AlphaError):
    def __init__(self):
        super().__init__(1)


class Alpha2(AlphaError):
    def __init__(self):
        super().__init__(2)


class Alpha3(AlphaError):
    def __init__(self):
        super().__init__(3)


class RMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(squared=False)


class LogRMSE(Metric):
    """Computes the rmse of two depth maps in log space"""

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0))
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred, gt):
        "Updates the internal states"
        pred = pred[gt > 0]
        gt = gt[gt > 0]
        self.sum += torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2)).sum()
        self.n += gt.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


class AbsoluteRelative(Metric):
    """Computes the absolute relative error between two depth maps."""

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0))
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred, gt):
        "Updates the internal states"
        pred = pred[gt > 0]
        gt = gt[gt > 0]
        self.sum += (torch.abs(gt - pred) / gt).sum()
        self.n += gt.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


class AbsoluteRelativeSquared(Metric):
    """Computes the squared absolute relative error between two depth maps."""

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0))
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred, gt):
        "Updates the internal states"
        pred = pred[gt > 0]
        gt = gt[gt > 0]
        self.sum += ((gt - pred) ** 2 / gt).sum()
        self.n += gt.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


class FunctionalMetrics:
    "Functional sample-by-sample version of the evaluation metrics"

    # Alpha (delta) metrics
    @staticmethod
    def alphaN(pred, gt, N):
        pred = pred[gt > 0]
        gt = gt[gt > 0]
        thresh = torch.max((gt / pred), (pred / gt))
        sum = (thresh < (1.25**N)).float().sum()
        num_elements = pred.numel()
        return sum / num_elements

    @staticmethod
    def alpha1(pred, gt):
        return FunctionalMetrics.alphaN(pred, gt, 1)

    @staticmethod
    def alpha2(pred, gt):
        return FunctionalMetrics.alphaN(pred, gt, 2)

    @staticmethod
    def alpha3(pred, gt):
        return FunctionalMetrics.alphaN(pred, gt, 3)

    # absrel
    @staticmethod
    def absrel(pred, gt):
        pred = pred[gt > 0]
        gt = gt[gt > 0]
        sum = torch.sum(torch.abs((gt - pred) / (gt)))
        num_elements = gt.numel()
        return sum / num_elements

    ### SSI metrics ###

    @staticmethod
    def ssi_alphaN(pred, gt, N, clip_value=None):
        if len(pred.shape) == 4:  # remove extra dim
            pred = pred.squeeze(dim=1)
            gt = gt.squeeze(dim=1)
        mask = gt > 0

        gt_disp = gt.clone()
        gt_disp[mask] = 1 / gt[mask]

        scale, shift = compute_scale_and_shift(pred, gt_disp, mask)
        pred_aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)

        if clip_value:
            clip_disp = 1 / clip_value
            pred_aligned[pred_aligned < clip_disp] = clip_disp

        pred_depth = 1.0 / pred_aligned 

        return FunctionalMetrics.alphaN(pred_depth, gt, N)

    @staticmethod
    def ssi_alpha1(pred, gt, clip_value=None):
        return FunctionalMetrics.ssi_alphaN(pred, gt, 1, clip_value)

    @staticmethod
    def ssi_alpha2(pred, gt, clip_value=None):
        return FunctionalMetrics.ssi_alphaN(pred, gt, 2, clip_value)

    @staticmethod
    def ssi_alpha3(pred, gt, clip_value=None):
        return FunctionalMetrics.ssi_alphaN(pred, gt, 3, clip_value)

    @staticmethod
    def ssi_absrel(pred, gt, clip_value=None):
        if len(pred.shape) == 4:  # remove extra dim
            pred = pred.squeeze(dim=1)
            gt = gt.squeeze(dim=1)
        mask = gt > 0

        gt_disp = gt.clone()
        gt_disp[mask] = 1 / gt[mask]

        scale, shift = compute_scale_and_shift(pred, gt_disp, mask)
        pred_aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)

        if clip_value:
            clip_disp = 1 / clip_value
            pred_aligned[pred_aligned < clip_disp] = clip_disp

        pred_depth = 1.0 / pred_aligned 
        # pred_depth = 1.0 / pred_aligned
        return FunctionalMetrics.absrel(pred_depth, gt)
