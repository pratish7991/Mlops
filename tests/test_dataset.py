# tests/test_dataset.py

import torch
from torch.utils.data import DataLoader
from src.datasets.preprocessed_dataset import PreprocessedDataset


def test_preprocessed_dataset_loading(tmp_path):
    dummy_data = [
        {
            "fingerprint": torch.randn(3, 128, 128),
            "left": torch.randn(1, 64, 64),
            "right": torch.randn(1, 64, 64),
            "label": 0
        }
    ]

    file_path = tmp_path / "dummy.pt"
    torch.save(dummy_data, file_path)

    dataset = PreprocessedDataset(str(file_path))

    assert len(dataset) == 1

    fingerprint, left, right, label = dataset[0]

    assert fingerprint.shape == (3, 128, 128)
    assert left.shape == (1, 64, 64)
    assert right.shape == (1, 64, 64)
    assert isinstance(label, int)

    # Test DataLoader compatibility
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))

    assert batch[0].shape == (1, 3, 128, 128)
