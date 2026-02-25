Multimodal Biometric Recognition – MLOps Architecture
Overview

This repository demonstrates a production-grade MLOps pipeline for multimodal biometric recognition using fingerprint and iris data.

The focus is on:

Scalable data pipelines

Modular PyTorch architecture

Parallel preprocessing

Reproducible training

CI/CD integration

Performance benchmarking

System Architecture

Raw Dataset
     ↓
Parallel Preprocessing (multiprocessing)
     ↓
Serialized Torch Dataset (.pt)
     ↓
DataLoader (multi-worker)
     ↓
Multimodal Model (MobileNetV2 + CNN branches)
     ↓
Training Loop (modular)
     ↓
Logging + Checkpoints

Design Decisions

PyTorch chosen for flexibility and scalability

Parallel preprocessing using ProcessPoolExecutor

Preprocessed dataset caching to reduce I/O bottlenecks

Config-driven training

Deterministic reproducibility via seed control

CI pipeline ensures code quality and correctness

Performance Engineering

Benchmarking DataLoader with varying num_workers

Comparison between raw image loading vs preprocessed tensor loading

Reduced I/O overhead via serialized dataset

GPU-compatible training pipeline

Scalability Considerations

Data pipeline easily extendable to distributed training

Can integrate Ray or DDP for multi-node scaling

Preprocessing pipeline can be adapted for cloud storage (Azure Blob)

Dockerized for deployment portability

Running the Project
# Preprocess dataset
python -m src.preprocessing.preprocess

# Train model
python src/main.py

CI/CD

GitHub Actions runs:

flake8 linting

black formatting check

pytest unit tests

Future Improvements

MLflow experiment tracking

Distributed training (DDP)

Model registry integration

Azure cloud deployment