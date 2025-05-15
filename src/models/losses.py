"xLoss functions"

import torch
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torchgeometry.losses import SSIM
from torch.nn import functional as F

from models.normalization import (
    max_over_dims,
    min_over_dims,
    normalize_batch_zero_one,
)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1e-3) -> None:
        super().__init__()
        self.SSIM = SSIM(11, reduction="mean")
        self.alpha = alpha

    def forward(self, a, b):
        huber = nn.functional.huber_loss(a, b)
        ssim = self.SSIM(a, b)
        return (self.alpha * huber) + ((1 - self.alpha) * ssim)


class GlobalMeanRemovedLoss(torch.nn.Module):
    def forward(self, _y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean_dims = tuple(range(len(_y.shape)))[1:]
        reshape_shape = [-1] + [1] * (len(y.shape) - 1)

        mean_y = y.mean(dim=mean_dims).reshape(reshape_shape)
        _mean_y = _y.mean(dim=mean_dims).reshape(reshape_shape)

        norm_y = y - mean_y
        _norm_y = _y - _mean_y
        error = norm_y - _norm_y
        error = torch.abs(error)

        return error.mean()


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.float()
        y = y.long()
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        # the full loss: ((1 - pt)^gamma) * -alpha  * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def focal_loss(
    alpha: Optional[Sequence] = None,
    gamma: float = 0.0,
    reduction: str = "mean",
    ignore_index: int = -100,
    device="cpu",
    dtype=torch.float32,
) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index,
    )
    return fl


############################ M I D A S #########################################


def ssi_trim_loss(pred, true, mask):
    """
    This function calculates the SSI trim loss between the predicted and true values.
    The SSI trim loss is calculated by taking the absolute difference between the true and predicted values,
    sorting them in ascending order, and then taking the top 80% of the sorted values.
    The sum of these top 80% values is divided by the total number of pixels in the image.
    The average of the loss per batch is then returned.

    Parameters
    ----------
    pred : torch.tensor
        The predicted tensor of shape (batch_size, channels, height, width).
    true : torch.tensor
        The true tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    loss : torch.tensor
        The average SSI trim loss over the batch.
    """
    residual = true - pred
    residual[mask] = 0
    batch_size, height, width = residual.shape
    num_pixels = height * width
    num_pixels_crop = int(0.8 * num_pixels)
    abs_residual = torch.abs(residual)
    flat_abs_residual = abs_residual.view(batch_size, -1)
    # Get an index of sorted abs_residual
    _, sorted_idx = torch.sort(flat_abs_residual.detach(), dim=1)
    # Get the top 80% of the sorted abs_residual
    top_80_idx = sorted_idx[:, :num_pixels_crop]
    top_80_abs_residual = torch.gather(flat_abs_residual, 1, top_80_idx)
    # Sum the top 80% of the sorted abs_residual
    sum_top_80_abs_residual = torch.sum(top_80_abs_residual, dim=1)
    # Divide by the number of pixels
    loss_per_batch = sum_top_80_abs_residual / (2 * num_pixels)
    # Average over the batch
    loss = torch.mean(loss_per_batch)
    return loss


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


class MidasLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based", disparity=False):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        # self.__data_loss = ssi_trim_loss
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.disparity = disparity
        if self.disparity:
            print("Midas loss created with disparity!")

    def forward(self, prediction, target):
        if len(prediction.shape) == 4:  # remove extra dim
            prediction = prediction.squeeze(dim=1)
            target = target.squeeze(dim=1)

        mask = target > 0
        if self.disparity:
            target = 1 / target
            target = normalize_batch_zero_one(target)

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target, mask
            )

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class MidasLossMedian(nn.Module):
    def __init__(
        self,
        alpha=0.5,
        scales=4,
        reduction="batch-based",
        disparity=False,
        sum_losses=True,
    ):
        super().__init__()

        self.__data_loss = self.trimmed_mae_loss
        # self.__data_loss = ssi_trim_loss
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
        self.disparity = disparity
        if self.disparity:
            print("Midas loss created with disparity!")
        self.sum_losses = sum_losses

    def forward(self, prediction, target):
        if len(prediction.shape) == 4:  # remove extra dim
            prediction = prediction.squeeze(dim=1)
            target = target.squeeze(dim=1)

        mask = target > 0
        if self.disparity:
            target = 1 / target
            target = normalize_batch_zero_one(target)

        self.__prediction_ssi = MidasLossMedian.normalize_prediction_robust(
            prediction, mask
        )
        target_ = MidasLossMedian.normalize_prediction_robust(target, mask)

        data_loss = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            reg_loss = self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )
            return (data_loss + reg_loss) if self.sum_losses else (data_loss, reg_loss)
        return data_loss

    def trimmed_mae_loss(self, prediction, target, mask, trim=0.2):
        M = torch.sum(mask, (1, 2))
        res = prediction - target

        res = res[mask.bool()].abs()

        trimmed, _ = torch.sort(res.view(-1), descending=False)[
            : int(len(res) * (1.0 - trim))
        ]

        return trimmed.sum() / (2 * M.sum())

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    @staticmethod
    def normalize_prediction_robust(target, mask):
        target = target.float()
        mask = mask.float()
        ssum = torch.sum(mask, (1, 2)).to(target)
        valid = (ssum > 0).bool()

        m = torch.zeros_like(ssum).to(target)
        s = torch.ones_like(ssum).to(target)

        m[valid] = torch.median(
            (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
        ).values
        target = target - m.view(-1, 1, 1)

        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

        return target / (s.view(-1, 1, 1))
