import hydra
from omegaconf import DictConfig, OmegaConf

from src.datasets.preprocessed_dataset import PreprocessedDataset
from src.models.multimodal_model import MultiModalModel
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    set_seed(cfg.reproducibility.seed)

    logger = setup_logger("logs")

    dataset = PreprocessedDataset(cfg.data.preprocessed_path)

    model = MultiModalModel(cfg.model.num_classes)

    if cfg.training.distributed:
        from src.utils.ddp import setup_ddp

        setup_ddp()

    trainer = Trainer(model, dataset, cfg, logger)
    trainer.train()

    if cfg.training.distributed:
        from src.utils.ddp import cleanup_ddp

        cleanup_ddp()


if __name__ == "__main__":
    main()
