import torch
import hydra
from omegaconf import DictConfig
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from src.models.multimodal_model import MultiModalModel


def load_image(path, size, grayscale=False):
    transform_list = [transforms.Resize(size)]

    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    image = Image.open(path)
    if not grayscale:
        image = image.convert("RGB")

    return transform(image).unsqueeze(0)  # Add batch dimension


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Hydra CLI overrides
    model_path = cfg.get("model_path")
    fingerprint_path = cfg.get("fingerprint_path")
    left_iris_path = cfg.get("left_iris_path")
    right_iris_path = cfg.get("right_iris_path")

    if not all([model_path, fingerprint_path, left_iris_path, right_iris_path]):
        raise ValueError(
            "Please provide model_path, fingerprint_path, left_iris_path, right_iris_path"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MultiModalModel(cfg.model.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load inputs
    fingerprint = load_image(
        fingerprint_path, tuple(cfg.data.fingerprint_size), grayscale=False
    ).to(device)

    left = load_image(left_iris_path, tuple(cfg.data.iris_size), grayscale=True).to(
        device
    )

    right = load_image(right_iris_path, tuple(cfg.data.iris_size), grayscale=True).to(
        device
    )

    # Inference
    with torch.no_grad():
        output = model(fingerprint, left, right)
        probs = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    print(f"\nPredicted Class: {prediction.item()}")
    print(f"Confidence: {confidence.item():.4f}")


if __name__ == "__main__":
    main()
