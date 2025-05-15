"""Some auxiliary functions to the training process"""

import torch

from models.normalization import normalize_batch_zero_one
from models.losses import compute_scale_and_shift


def masked_loss(task_pred, task_true, true, mask_feature_index, task, loss):
    feat_mask = torch.ones_like(task_true).bool()
    nan_mask = ~task_true.isnan()
    if task.mask_feature:
        feat_mask = true[:, mask_feature_index, ...]
        feat_mask = torch.unsqueeze(feat_mask, 1)
        feat_mask = feat_mask / 255
        feat_mask = feat_mask.bool()

    mask = nan_mask & feat_mask

    batch_loss = 0
    for m, t, p in zip(mask, task_true, task_pred):
        batch_loss += loss(p[m], t[m])
    batch_loss /= true.shape[0]
    return batch_loss


def class_loss_preproc(task, task_pred, task_true):
    assert task_true.max() <= (
        task.num_classes
    ), "class gt larger than number of classes"

    task_pred = task_pred.swapaxes(-1, -3)
    task_pred = task_pred.reshape(-1, task.num_classes)
    task_true = task_true.flatten().long()

    ignore_mask = task_true != -1
    task_pred = task_pred[ignore_mask]
    task_true = task_true[ignore_mask]
    return task_pred, task_true


def masked_metrics(task_pred, task_true, true, mask_feature_index, task, metrics):
    nan_mask = ~task_true.isnan()
    feat_mask = torch.ones_like(task_true).bool()
    if task.mask_feature:
        feat_mask = true[:, mask_feature_index, ...]
        feat_mask = torch.unsqueeze(feat_mask, 1)
        feat_mask = feat_mask / 255
        feat_mask = feat_mask.bool()

    mask = nan_mask & feat_mask

    _metrics = {}
    for metric_name, metric in metrics:
        batch_metric = 0
        for t, p, m in zip(task_true, task_pred, mask):
            batch_metric += metric(p[m], t[m])
        batch_metric /= task_true.shape[0]
        _metrics[f"{task.name}_{metric_name}"] = batch_metric
    return _metrics


def prepare_to_metrics(task_pred, task_true):

    if len(task_pred.shape) == 5:  # remove extra dim
        task_pred = task_pred.squeeze(dim=(1)).squeeze(dim=(1))
        task_true = task_true.squeeze(dim=(1)).squeeze(dim=(1))

    gt_disp = task_true.clone()
    gt_disp[task_true > 0] = 1 / task_true[task_true > 0]

    scale, shift = compute_scale_and_shift(task_pred, gt_disp, gt_disp > 0)
    task_pred = scale.view(-1, 1, 1) * task_pred + shift.view(-1, 1, 1)

    task_pred[task_pred != 0] = 1 / task_pred[task_pred != 0]

    return task_pred, task_true
