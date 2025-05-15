"Definition of a model task"

import torch
import torchmetrics
import torchmetrics.classification

from models.losses import CombinedLoss, FocalLoss


def get_task(task: str):
    "Gets a task class from a string"
    available_tasks = {
        "dense_regression": DenseRegression,
        "dense_classification": DenseClassification,
    }
    return available_tasks[task.lower()]


class Task:
    """A network task"""

    def __init__(self, decoder, features, channels, **args) -> None:
        """"""

        self.decoder = decoder  # class of Decoder

        self.features = features  #  [label_feats : str]

        self.channels = channels

        self.train_on_disparity = args.get("train_on_disparity", False)

        self.decoder_args = args.get("decoder_args", {})

        self.disp_metrics = args.get("disp_metrics", True)

    def compute_metric(self, pred, true):
        results = {}
        for metric in self.metric:
            results[metric] = self.metric[metric](pred, true)
        return results

    def compute_loss(self, pred, true):
        """Computes loss for a batch of predicitons"""
        return self.loss(pred, true)


class DenseRegression(Task):
    "Simple dense regression task"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        # Default loss if none specified
        self.loss = args.get("loss") or CombinedLoss()
        # Metrics
        self.metric = args.get("metrics") or {
            "rmse": torchmetrics.MeanSquaredError(squared=False)
        }

        self.name = args.get("name") or "dense_regression"

        self.mask_feature = args["mask_feature"]


class DenseClassification(Task):
    "Simple dense classification task"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        # Default loss if none specified
        # self.loss = args.get("loss") or FocalLoss()
        self.loss = args.get("loss") or torch.nn.CrossEntropyLoss(ignore_index=-1)
        # Metrics
        self.metric = args.get("metrics") or {}

        self.name = args.get("name") or "dense_classification"

        self.mask_feature = args["mask_feature"]

        self.num_classes = self.channels

    def compute_loss(self, pred, true):
        "Flattens the tensors and compute the loss"
        pred = pred.reshape(-1, self.num_classes)
        true = true.flatten()
        return self.loss(pred.float(), true)
