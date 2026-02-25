import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["fingerprint"],
            item["left"],
            item["right"],
            item["label"]
        )
