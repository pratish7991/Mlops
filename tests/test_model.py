# tests/test_model.py

import torch
from src.models.multimodal_model import MultiModalModel


def test_model_forward_and_backward():
    model = MultiModalModel(num_classes=45)

    fingerprint = torch.randn(2, 3, 128, 128)
    left = torch.randn(2, 1, 64, 64)
    right = torch.randn(2, 1, 64, 64)

    output = model(fingerprint, left, right)

    assert output.shape == (2, 45)

    loss = output.mean()
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
