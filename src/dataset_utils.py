"Utility functions for dataset preparation and loading"

import multiprocessing

from datasets import get_dataloader
from datasets.mixed_dataset import MixedDataset
from default_paths import DEFAULT_SPLIT_FILES, DEFAULT_DATASET_ROOT


def check_dataset_paths(datasets, dataset_roots, splits, root_config=None):
    """Check if the dataset paths are valid"""
    number_of_datasets = len(datasets)
    if (dataset_roots is not None and len(dataset_roots) == number_of_datasets) and (
        splits is not None and len(splits) == number_of_datasets
    ):
        # all dataset paths are provided, returning
        return dataset_roots, splits

    # Some problem with the paths, using defaults
    dataset_roots = []
    splits = []
    for dataset in datasets:
        dataset_roots.append(root_config[dataset.lower()])
        splits.append(DEFAULT_SPLIT_FILES[dataset.lower()])
    return dataset_roots, splits


def prepare_dataset(datasets, target_size=None, **args):
    """Prepares datasets for train, validation and test"""

    dataset_roots = args.get("dataset_root")  # Retrocomp
    splits = args.get("split_json")
    batch_size = args["batch_size"]
    num_workers = args.get("num_workers", multiprocessing.cpu_count())

    train_loaders = []
    val_loaders = []
    test_loaders = []

    dataset_roots, splits = check_dataset_paths(
        datasets, dataset_roots, splits, root_config=DEFAULT_DATASET_ROOT
        )

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(dataset_roots, str):
        dataset_roots = [dataset_roots]
    if isinstance(splits, str):
        splits = [splits]

    # __________________________________________________________________________
    if args.get("train", False):
        print("Preparing train datasets...")
        for dataset, dataset_root, split_json in zip(datasets, dataset_roots, splits):
            Dataset = get_dataloader(dataset)
            args["split_json"] = split_json
            train_dataset = Dataset(
                dataset_root=dataset_root,
                split="train",
                target_size=target_size,
                **args,
            )
            train_loaders.append(train_dataset)
    # __________________________________________________________________________
    if args.get("validation", False):
        print("Preparing validation datasets...")
        for dataset, dataset_root, split_json in zip(datasets, dataset_roots, splits):
            Dataset = get_dataloader(dataset)
            args["split_json"] = split_json

            val_dataset = Dataset(
                dataset_root=dataset_root,
                split="validation",
                target_size=target_size,
                **args,
            )
            val_loaders.append(val_dataset)
    # __________________________________________________________________________
    if args.get("test", False):
        print("Preparing test dataset...")
        for dataset, dataset_root, split_json in zip(datasets, dataset_roots, splits):
            Dataset = get_dataloader(dataset)
            args["split_json"] = split_json

            test_dataset = Dataset(
                dataset_root=dataset_root,
                split="test",
                target_size=target_size,
                **args,
            )
            test_loaders.append(test_dataset)

    return (
        MixedDataset(datasets=train_loaders).build_dataloader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        MixedDataset(datasets=val_loaders).build_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        (
            MixedDataset(datasets=test_loaders).build_dataloader(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            if len(test_loaders) > 0
            else None
        ),
    )
