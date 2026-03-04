import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.preprocessing.preprocess import preprocess_dataset
from src.datasets.preprocessed_dataset import PreprocessedDataset
from src.models.multimodal_model import MultiModalModel
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Set reproducibility seed
    set_seed(cfg.reproducibility.seed)

    # Logger
    logger = setup_logger("logs")

    data_path = cfg.data.preprocessed_path

    # If dataset not found, run preprocessing
    if not os.path.exists(data_path):
        print("Processed dataset not found. Running preprocessing...")

        preprocess_dataset(
            base_path=cfg.data.base_path,
            num_people=cfg.data.num_people,
            fingerprint_size=tuple(cfg.data.fingerprint_size),
            iris_size=tuple(cfg.data.iris_size),
            output_file=data_path,
            num_workers=cfg.data.num_workers,
        )

    # Load dataset
    dataset = PreprocessedDataset(data_path)

    # Initialize model
    model = MultiModalModel(cfg.model.num_classes)

    # Optional Distributed Training
    if cfg.training.distributed:
        from src.utils.ddp import setup_ddp
        setup_ddp()

    # Trainer
    trainer = Trainer(model, dataset, cfg, logger)
    trainer.train()

    # Cleanup distributed
    if cfg.training.distributed:
        from src.utils.ddp import cleanup_ddp
        cleanup_ddp()


if __name__ == "__main__":
    main()
