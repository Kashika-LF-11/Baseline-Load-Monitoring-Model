# SageMaker Training Pipeline

A production-grade modular training package designed to run inside AWS SageMaker training containers. This project implements a complete AutoEncoder training pipeline with ONNX export, following clean architecture principles and the SageMaker training contract.

## Overview

The training system is built around four independent pipeline modules orchestrated by `train.py`:

1. **Ingestion** — Load raw 2D signals from SageMaker input channels
2. **Preprocessing (M1)** — Convert 2D time-series into 3D sequences, validate schema, compute statistics
3. **Model Init (M2)** — This is a central model initialization module used by training, calibration, export modeules. 
4. **Model Training(M3)** — Initialize, train, and checkpoint a PyTorch LSTM AutoEncoder
5. **Calibration (M4)** — Compute validation metrics and residual error distribution
6. **Export (M5)** — Convert trained model to ONNX for inference deployment

## Architecture

```
train.py (orchestrator)
├── [M1] modules/preprocessing.py   → 2D → 3D sequences → save feature_config.json
├── [M2] modules/model_init.py      → Define LSTM AutoEncoder, save model_config.json
├── [M3] modules/trainer.py         → Training loop → model.keras saved
├── [M4] modules/calibration.py     → Reload, validate, compute metrics → metadata.json
└── [M5] modules/export_onnx.py     → Tensorflow → ONNX 
```

## Module Details

### M2: Preprocessing (`modules/preprocessing.py`)
**Responsibility:** Convert 2D signals into 3D training sequences.

- Slides a window of size `sequence_length` across time-series data
- Creates sequences of shape `(num_sequences, sequence_length, num_features)`
- Saves `feature_config.json` with metadata for downstream inference
- Raises error if data length < sequence_length

**Example:** If data is shape (1000, 5) and sequence_length=30, output is (971, 30, 5).

### M3: Model & Training (`modules/model_init.py` + `modules/trainer.py`)

# SageMaker training (LSTM AutoEncoder)

## Purpose

Lightweight, modular training code intended to run inside an AWS SageMaker training container or locally for development. The pipeline trains an LSTM AutoEncoder, performs calibration, and exports a production ONNX model.

## Layout (short)

- `train.py` — orchestrator
- `config/` — defaults, paths, logging
- `modules/` — pipeline implementations (`ingestion.py`, `preprocessing.py`, `model_init.py`, `trainer.py`, `calibration.py`, `export_onnx.py`)
- `tests/local_test_run.py` — local SageMaker layout simulator
- `requirements.txt`, `Dockerfile`, `README.md`


## SageMaker contract (quick)

- Input data directory: `/opt/ml/input/data/train` (CSV file)
- Model output: `/opt/ml/model/model.onnx`
- Intermediate artifacts: `/opt/ml/output/intermediate/`
    - `models/model.kears`
    - `artifacts/metadata.json`

## Config & Defaults

Defaults live in `config/constants.py`. Example hyperparameters:

```json
{
  "sequence_length": 10,
  "epochs": 50,
  "batch_size": 64,
  "dim_weights": [0.8, 0.05, 0.05, 0.05, 0.025]
}
```

## Where to look in code

- Preprocessing: `modules/preprocessing.py` (converts 2D → 3D sequences)
- Model: `modules/model_init.py`
- Trainer & checkpoints: `modules/trainer.py`
- Calibration: `modules/calibration.py`
- ONNX export: `modules/export_onnx.py`

## Artifacts produced

- `metadata.json` — calibration residual statistics
- `model.keras` — best model saved (used to save weights from)
- `model.onnx` — exported ONNX model
