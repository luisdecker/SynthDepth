"Union of several datasets"

from datasets.dataset import Dataset
from torch.utils.data import ConcatDataset


class MixedDataset(Dataset):
    """"""

    def __init__(self, **args):
        self.datasets = args.get("datasets")
        self.concat_ds = ConcatDataset(self.datasets)
        self.size = None

    def __len__(self):
        "Get number of samples of dataset"
        if self.size is not None:
            return self.size
        self.size = len(self.concat_ds)
        return self.size

    def __getitem__(self, idx):
        return self.concat_ds[idx]
