import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# import mlflow.pytorch
from datetime import datetime
import mlflow
import time
import platform
from omegaconf import OmegaConf


class Trainer:
    def __init__(self, model, dataset, config, logger):
        self.config = config
        self.logger = logger

        # Device handling (safe)
        if config["training"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.dataset = dataset

        self.dataloader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=config["model"]["learning_rate"]
        )

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"exp_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config snapshot

        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))

    def train(self):


        epochs = self.config["training"]["epochs"]

        mlflow.set_experiment("multimodal_biometric")

        # Always close previous run
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # Start run FIRST
        run = mlflow.start_run()

        try:
            # ========== LOG PARAMS ==========
            dataset_size = len(self.dataset)
            device_name = str(self.device)

            mlflow.log_param("dataset_size", dataset_size)
            mlflow.log_param("device", device_name)
            mlflow.log_param("platform", platform.platform())
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", self.config["data"]["batch_size"])
            mlflow.log_param("learning_rate", self.config["model"]["learning_rate"])
            mlflow.log_param("num_workers", self.config["data"]["num_workers"])

            # ========== DATALOADER TIMING ==========
            start_time = time.time()
            for _ in self.dataloader:
                break
            first_batch_time = time.time() - start_time
            mlflow.log_metric("first_batch_time_sec", first_batch_time)

            # ========== TRAINING LOOP ==========
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0

                for fingerprint, left, right, labels in self.dataloader:
                    fingerprint = fingerprint.to(self.device)
                    left = left.to(self.device)
                    right = right.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(fingerprint, left, right)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(self.dataloader)

                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

                mlflow.log_metric("loss", avg_loss, step=epoch)

            # ========== SAVE MODEL ==========
            model_path = os.path.join(self.run_dir, "model.pt")
            torch.save(self.model.state_dict(), model_path)

            mlflow.pytorch.log_model(self.model, artifact_path="model")

            # Register model under SAME RUN
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="MultimodalBiometricModel",
            )

            self.logger.info(f"Model saved to {model_path}")

        finally:
            mlflow.end_run()
