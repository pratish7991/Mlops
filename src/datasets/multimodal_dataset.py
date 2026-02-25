import os
from typing import Tuple, List


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MultiModalDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        num_people: int,
        fingerprint_size: Tuple[int, int],
        iris_size: Tuple[int, int],
    ):
        self.base_path = base_path
        self.num_people = num_people

        self.fingerprint_transform = transforms.Compose(
            [transforms.Resize(fingerprint_size), transforms.ToTensor()]
        )

        self.iris_transform = transforms.Compose(
            [
                transforms.Resize(iris_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        self.samples = self._collect_samples()

    def _collect_samples(self) -> List:
        samples = []

        for person_id in range(1, self.num_people + 1):
            person_path = os.path.join(self.base_path, str(person_id))

            if not os.path.exists(person_path):
                continue

            fingerprint_dir = os.path.join(person_path, "Fingerprint")
            left_dir = os.path.join(person_path, "left")
            right_dir = os.path.join(person_path, "right")

            fingerprint_file = self._get_first_bmp(fingerprint_dir)
            left_file = self._get_first_bmp(left_dir)
            right_file = self._get_first_bmp(right_dir)

            if fingerprint_file and left_file and right_file:
                samples.append((fingerprint_file, left_file, right_file, person_id - 1))

        if len(samples) == 0:
            raise ValueError("No valid samples found.")

        return samples

    def _get_first_bmp(self, directory: str):
        if not os.path.exists(directory):
            return None
        for file in os.listdir(directory):
            if file.lower().endswith(".bmp"):
                return os.path.join(directory, file)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fingerprint_path, left_path, right_path, label = self.samples[idx]

        fingerprint = self.fingerprint_transform(
            Image.open(fingerprint_path).convert("RGB")
        )

        left_iris = self.iris_transform(Image.open(left_path))

        right_iris = self.iris_transform(Image.open(right_path))

        return fingerprint, left_iris, right_iris, label
