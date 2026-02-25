import os
import torch
from torchvision import transforms
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List


def process_person(
    args: Tuple[str, int, Tuple[int, int], Tuple[int, int]]
):
    base_path, person_id, fingerprint_size, iris_size = args

    person_path = os.path.join(base_path, str(person_id))

    fingerprint_dir = os.path.join(person_path, "Fingerprint")
    left_dir = os.path.join(person_path, "left")
    right_dir = os.path.join(person_path, "right")

    def get_first_bmp(directory):
        if not os.path.exists(directory):
            return None
        for f in os.listdir(directory):
            if f.lower().endswith(".bmp"):
                return os.path.join(directory, f)
        return None

    fingerprint_file = get_first_bmp(fingerprint_dir)
    left_file = get_first_bmp(left_dir)
    right_file = get_first_bmp(right_dir)

    if not (fingerprint_file and left_file and right_file):
        return None

    fingerprint_transform = transforms.Compose([
        transforms.Resize(fingerprint_size),
        transforms.ToTensor()
    ])

    iris_transform = transforms.Compose([
        transforms.Resize(iris_size),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

    fingerprint = fingerprint_transform(
        Image.open(fingerprint_file).convert("RGB")
    )

    left = iris_transform(Image.open(left_file))
    right = iris_transform(Image.open(right_file))

    return {
        "fingerprint": fingerprint,
        "left": left,
        "right": right,
        "label": person_id - 1
    }


def preprocess_dataset(
    base_path: str,
    num_people: int,
    fingerprint_size: Tuple[int, int],
    iris_size: Tuple[int, int],
    output_file: str,
    num_workers: int = 4
):

    print("Starting parallel preprocessing...")

    args_list = [
        (base_path, pid, fingerprint_size, iris_size)
        for pid in range(1, num_people + 1)
    ]

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_person, args_list):
            if result is not None:
                results.append(result)

    torch.save(results, output_file)

    print(f"Saved preprocessed dataset to {output_file}")



if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    preprocess_dataset(
        base_path=config["data"]["base_path"],
        num_people=config["data"]["num_people"],
        fingerprint_size=tuple(config["data"]["fingerprint_size"]),
        iris_size=tuple(config["data"]["iris_size"]),
        output_file=config["data"]["preprocessed_path"],
        num_workers=config["data"]["num_workers"],
    )
