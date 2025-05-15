"Definition a abstract model and implementation of common model functions"

from pytorch_lightning import LightningModule
import torch
import torchmetrics
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from models.losses import compute_scale_and_shift
from models.training_utils import (
    class_loss_preproc,
    masked_loss,
    masked_metrics,
    prepare_to_metrics,
)


class Model(LightningModule):
    """Abstract Model"""

    def __init__(self, **args):
        super().__init__()

        # Save args in a atributte to future recover

        self.save_hyperparameters()

        # Configure each one of the models tasks
        self.tasks = args["tasks"]

        metrics = {}
        for task in self.tasks:
            metrics[task.name] = torch.nn.ModuleDict(task.metric)
            metrics[task.name]["loss"] = task.loss

        self.metrics = torch.nn.ModuleDict(metrics)

        self.features = args["features"]

        self.savepath = args.get("savepath")

        self.train_loss = torchmetrics.MeanMetric()
        self.val_outputs = []
        self.num_loaders = args.get("num_loaders", 1)

        self.lr = args.get("lr", 1e-3)
        self.scheduler_steps = args.get("scheduler_steps", [20, 80])
        self.scheduler_name = args.get("scheduler_name", "cossine_restart")
        self.epochs = args.get("epochs")
        self.pretrained_encoder = args.get("pretrained_encoder", False) | args.get(
            "start_frozen", False
        )
        self.encoder_lr = args.get("encoder_lr", self.lr)

    def expand_shape(self, x):
        """Expands all tensors in the list x to have the same shape in the
        channels dimension.

        x's shape is expected to be (batch, channels, ...)
        """
        largest_feature_size = max([t.channels for t in self.tasks])
        for i, _x in enumerate(x):
            if _x.shape[1] == largest_feature_size:
                continue
            expanded_shape = list(_x.shape)
            expanded_shape[1] = largest_feature_size
            expanded = torch.zeros(expanded_shape).to(_x.device)
            expanded[:, : _x.shape[1], ...] = _x
            x[i] = expanded
        return x

    # Torchlightning steps ====================================================
    def training_step(self, batch, batch_idx):
        """One step of training"""
        # if self.num_loaders == 1:
        #     batch = [batch]
        loss = 0

        x, y = batch
        _y = self.forward(x)

        loss = self.compute_loss(_y, y)
        self.log("train_step_loss", loss, sync_dist=True, prog_bar=True)
        self.train_loss.update(loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=None):

        x, y = batch

        _y = self.forward(x)

        metrics = self.compute_metric(_y, y)
        metrics = {("val_step_" + name): val for name, val in metrics.items()}
        self.val_outputs.append(metrics)
        # self.log_dict(metrics, logger=True)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self, outputs=None):
        if outputs:
            _metrics = {}

            for task in self.tasks:
                if outputs and isinstance(outputs[0], dict):
                    task_losses = [
                        output[f"val_step_{task.name}_loss"] for output in outputs
                    ]
                    _metrics[f"test_{task.name}_loss"] = torch.stack(task_losses).mean()

                for metric_name, metric in self.metrics[task.name].items():
                    if hasattr(metric, "compute"):
                        _metrics[f"test_{task.name}_{metric_name}"] = (
                            metric.compute().mean()
                        )
                        metric.reset()
            self.log_dict(_metrics, sync_dist=True)

    def on_train_epoch_end(self):
        """"""

        with torch.no_grad():
            _metrics = {}
            _metrics["train_loss_epoch"] = self.train_loss.compute()
            self.train_loss.reset()

            self.log_dict(_metrics, logger=True, prog_bar=True, sync_dist=True)

            # Log some images
            dataloader = self.trainer.val_dataloaders
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
            dataset = dataloader.dataset
            samples = (dataset[0], dataset[len(dataset) // 2], dataset[-1])
            for i, sample in enumerate(samples):
                image, gt = sample
                gt = gt[0]
                pred = self.forward(image.unsqueeze(0).float().to(self.device))[0, 0]
                self.plot_images(f"Sample {i}", gt, pred, self.current_epoch)

    def on_validation_epoch_end(self):
        """"""
        _metrics = {}

        for task in self.tasks:
            if self.val_outputs and isinstance(self.val_outputs[0], dict):
                task_losses = [
                    output[f"val_step_{task.name}_loss"] for output in self.val_outputs
                ]
                _metrics[f"val_{task.name}_loss"] = torch.stack(task_losses).mean()

            for metric_name, metric in self.metrics[task.name].items():
                if hasattr(metric, "compute"):
                    _metrics[f"val_{task.name}_{metric_name}"] = metric.compute().mean()
                    metric.reset()
        self.log_dict(_metrics, logger=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    def _get_optimizer(self):
        if self.encoder_lr:
            print(f"Setting encoder lr to {self.encoder_lr}")
            param_groups = [
                {"params": self.decoders.parameters()},
                {
                    "params": self.encoder.parameters(),
                    "lr": self.encoder_lr,
                    "name": "encoder",
                },
            ]
            return torch.optim.AdamW(param_groups, lr=self.lr)
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _get_scheduler(self, optimizer):
        return {
            "cossine": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, verbose=True
            ),
            "cossine_restart": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=100, T_mult=2, verbose=True
            ),
        }[self.scheduler_name]

    def compute_loss(self, pred, true):
        """Computes loss for all the tasks"""

        # Tasks must be in order
        total_loss = 0
        for task_index, task in enumerate(self.tasks):
            label_idx = [self.features[1].index(feat) for feat in task.features]

            # Gets data from the specified task
            task_pred = pred[:, [task_index], ...]

            # Remove any expanded data
            task_pred = task_pred[:, :, : task.channels, ...]
            task_true = true[:, label_idx, ...]

            has_nans = task_true.isnan().any()
            if task.mask_feature or has_nans:

                batch_loss = masked_loss(
                    task_pred,
                    task_true,
                    true,
                    self.features[1].index(task.mask_feature),
                    task,
                    self.metrics[task.name]["loss"],
                )
                total_loss += batch_loss
                continue

            task_pred = task_pred.squeeze(dim=1)
            task_true = task_true.squeeze(dim=1)

            if hasattr(task, "num_classes"):
                task_pred, task_true = class_loss_preproc(task, task_pred, task_true)

            loss = self.metrics[task.name]["loss"](task_pred, task_true)
            if isinstance(loss, tuple):
                data_loss, reg_loss = loss
                self.log(
                    "train_step_loss/data_loss",
                    data_loss,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    "train_step_loss/reg_loss", reg_loss, prog_bar=True, sync_dist=True
                )
                total_loss += data_loss + reg_loss
                continue
            total_loss += self.metrics[task.name]["loss"](task_pred, task_true)

        return total_loss

    def compute_metric(self, pred, true):
        """Computes metric for all the tasks"""
        with torch.no_grad():
            # Tasks must be in order
            metrics = {}
            for task_index, task in enumerate(self.tasks):
                label_idx = [self.features[1].index(feat) for feat in task.features]
                task_pred = pred[:, [task_index], ...]
                task_pred = task_pred[:, :, : task.channels, ...]
                task_true = true[:, label_idx, ...]

                if task.train_on_disparity:
                    # Reconstruct the image for metric computing
                    task_true = 1.0 / task_true

                has_nans = task_true.isnan().any()
                if task.mask_feature or has_nans:
                    metrics.update(
                        masked_metrics(
                            task_pred,
                            task_true,
                            true,
                            self.features[1].index(task.mask_feature),
                            task,
                            self.metrics[task.name].items(),
                        )
                    )
                    continue

                task_metrics = self.metrics[task.name]
                _metrics = {}
                for metric_name, metric in task_metrics.items():
                    metric_true = task_true.clone()
                    metric_pred = task_pred.clone()

                    if metric_name == "loss" and hasattr(task, "num_classes"):
                        _metrics[metric_name] = metric(
                            metric_pred.squeeze(dim=1)
                            .swapaxes(-1, -3)
                            .reshape(-1, task.num_classes)
                            .float(),
                            metric_true.squeeze(dim=1).flatten().long(),
                        )
                        continue

                    if task.disp_metrics and not metric_name == "loss":
                        metric_pred, metric_true = prepare_to_metrics(
                            metric_pred, metric_true
                        )

                    if metric_name == "loss":
                        computed_loss = metric(
                            metric_pred.squeeze(dim=1), metric_true.squeeze(dim=1)
                        )
                        if isinstance(computed_loss, tuple):
                            _metrics["data_loss"] = computed_loss[0]
                            _metrics["reg_loss"] = computed_loss[1]
                            _metrics["loss"] = computed_loss[0] + computed_loss[1]
                            continue
                        _metrics["loss"] = computed_loss
                        continue

                    _metrics[metric_name] = metric(
                        metric_pred.squeeze(dim=1), metric_true.squeeze(dim=1)
                    )

                for metric, value in _metrics.items():
                    name = f"{task.name}_{metric}"
                    metrics[name] = value

        return metrics

    def to_gpu(self, gpu=0):  # I need to remove this
        self = self.cuda(gpu)
        [decoder.cuda(gpu) for decoder in self.decoders]
        return self

    def plot_images(self, label, gt, pred, step):
        pred = pred.squeeze(dim=1)
        gt = gt.to(pred)
        gt[gt > 0] = 1 / gt[gt > 0]
        scale, shift = compute_scale_and_shift(pred[0], gt, gt > 0)
        pred = scale.view(-1, 1, 1) * pred[0] + shift.view(-1, 1, 1)

        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()[0]

        min_depth = gt.min()
        max_depth = gt.max()

        gt = (gt - min_depth) / (max_depth - min_depth)
        gt = plt.cm.turbo_r(gt[0])[..., :3]
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()

        pred = (pred - min_depth) / (max_depth - min_depth)
        pred = plt.cm.turbo_r(pred)[..., :3]
        pred = torch.from_numpy(pred).permute(2, 0, 1).float()

        grid = vutils.make_grid([gt, pred], nrow=2, padding=5)

        tb = self.logger.experiment
        tb.add_image(f"GT x Pred /({label})", grid, step)
