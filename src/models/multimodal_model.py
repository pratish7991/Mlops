import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

class IrisBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class MultiModalModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Fingerprint branch
        self.fingerprint_model = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )
        self.fingerprint_model.classifier = nn.Identity()

        # Iris shared branch
        self.iris_branch = IrisBranch()

        self.classifier = nn.Sequential(
            nn.Linear(1280 + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, fingerprint, left_iris, right_iris):
        fingerprint_feat = self.fingerprint_model(fingerprint)
        left_feat = self.iris_branch(left_iris)
        right_feat = self.iris_branch(right_iris)

        combined = torch.cat([fingerprint_feat, left_feat, right_feat], dim=1)

        return self.classifier(combined)
