""""""

import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)

from callbacks import UnfreezeEncoder
from dataset_utils import prepare_dataset
from datasets.kitti import Kitti
from datasets.nyu import NYUDepthV2
from default_paths import DEFAULT_DATASET_ROOT, DEFAULT_SPLIT_FILES
from eval_utils import evaluate_dataset, get_model
from file_utils import save_json
from metrics import FunctionalMetrics
from args_utils import read_args
from models.convNext import ConvNext

torch.set_float32_matmul_precision("medium")


def freeze_encoder(model):
    "Freezes model encoder"
    for param in model.encoder.parameters():
        param.requires_grad = False


def train(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    DEVICE = args["gpu"]
    DEVICE = [DEVICE] if isinstance(DEVICE, int) else DEVICE

    # Generate log path

    logpath = os.path.join("logs", args["expname"])

    # logpath = os.path.join(logpath, f"_{get_last_exp(logpath)}")
    os.makedirs(logpath, exist_ok=True)

    # Saves the args in the path
    args_path = os.path.join(logpath, "args.json")
    save_json(args_path, args["input_args"])

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataset(
        datasets=args["dataset"], train=True, validation=True, **args
    )

    # Try to train some network
    model = ConvNext(
        num_loaders=1,
        **args,
    )

    if pretrained_path := args.get("pretrained_path"):
        # Load model weigths only
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint["state_dict"])

    if args.get("start_frozen"):

        freeze = bool(args.get("start_frozen"))

        # Check if have to freeze in case of continuing a training
        if ckpt_path := args["ckpt_path"]:
            ckpt = torch.load(ckpt_path)
            last_epoch = ckpt["epoch"]
            freeze = last_epoch < args["unfreeze_epoch"]
            del ckpt

        if freeze:
            print("===================================")
            print("Warning: Encoder starting frozen!!!")
            print("===================================")
            freeze_encoder(model)

    callbacks = [
        RichModelSummary(max_depth=5),
        RichProgressBar(refresh_rate=1, leave=True),
        LearningRateMonitor(logging_interval="step"),
        StochasticWeightAveraging(
            swa_lrs=[args.get("lr", 1e-3) * 1e-2, args.get("encoder_lr", 1e-5) * 1e-2]
        ),
        ModelCheckpoint(verbose=True, every_n_epochs=1),
    ]
    if epoch := args.get("unfreeze_epoch", None):
        callbacks.append(UnfreezeEncoder(epoch))

    trainer_args = {
        "log_every_n_steps": 10,
        "check_val_every_n_epoch": 5,
        "num_sanity_val_steps": 4,
        "max_epochs": args["epochs"],
        "accelerator": "gpu",
        "gradient_clip_val": 0.5,
        "default_root_dir": logpath,
        "callbacks": callbacks,
        "logger": True,
        "enable_checkpointing": True,
        "accumulate_grad_batches": args.get("acummulate_batches", 1),
        "precision": "16-mixed",
    }

    if args["slurm"]:
        trainer_args.update(
            {
                "devices": "auto",
                "num_nodes": int(os.getenv("SLURM_JOB_NUM_NODES", "1")),
            }
        )
    else:
        trainer_args.update(
            {
                "devices": DEVICE,
                "num_nodes": 1,
            }
        )

    trainer = pl.Trainer(**trainer_args)

    # model = torch.compile(model)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args["ckpt_path"],
    )

    print("DONE!")


def test(args):
    """Test a model in KITTI and NYU datasets"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    pd.options.display.float_format = "{:.3f}".format
    # Load model
    model = get_model(args)

    # Load Datasets
    print("[EVALUATION] Loading Kitti Validation Dataset!")
    args["split_json"] = DEFAULT_SPLIT_FILES["kitti"]
    args["dataset_root"] = DEFAULT_DATASET_ROOT["kitti"]
    kitti_val_ds = Kitti(split="validation", **args)

    print("[EVALUATION] Loading NYU Validation Dataset!")
    args["split_json"] = DEFAULT_SPLIT_FILES["nyu"]
    args["dataset_root"] = DEFAULT_DATASET_ROOT["nyu"]
    nyu_val_ds = NYUDepthV2(split="validation", **args)

    # Load metrics
    metrics_depth = {
        "absrel": FunctionalMetrics.absrel,
        "delta1": FunctionalMetrics.alpha1,
        "delta2": FunctionalMetrics.alpha2,
        "delta3": FunctionalMetrics.alpha3,
    }

    # Compute metrics for NYU

    print("[EVALUATION] Computing metrics for NYU Depth V2 dataset!")
    results_nyu = evaluate_dataset(nyu_val_ds, model, metrics_depth, max_depth=10)
    print()
    print(pd.DataFrame([r for r in results_nyu.values()]).astype(float).describe())

    print()
    print("[EVALUATION] Computing metrics for KITTI!")
    results_kitti = evaluate_dataset(kitti_val_ds, model, metrics_depth, max_depth=80)
    print()
    print(pd.DataFrame([r for r in results_kitti.values()]).astype(float).describe())


if __name__ == "__main__":
    args = read_args()
    if args["evaluate_path"]:
        test(args)
    else:
        train(args)
    print("Done!")
