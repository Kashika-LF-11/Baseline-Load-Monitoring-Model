"""
train.py
--------
SageMaker training entrypoint.

This script orchestrates the full pipeline:

1) Data ingestion
2) Preprocessing
3) Model initialization
4) Training (produces model.keras)
5) Calibration (produces metadata.json)
6) ONNX export (produces model.onnx)

"""

import json
import random
import pandas as pd
import numpy as np
import tensorflow as tf

from config import logging_config, paths
from config.constants import DEFAULT_SEQUENCE_LENGTH, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_DIM_WEIGHTS
from modules import preprocessing, model_init, trainer, calibration, export_onnx

logger = logging_config.configure_logging()
# data_type_name = 'PROGRAM_SCOPED_TOOL_ROTATION' 

# ---------------------------------------------------------------------
# 1) Load CSV directly from SageMaker channel
# ---------------------------------------------------------------------
def load_training_csv() -> np.ndarray:
    train_dir = paths.input_data_dir()
    csv_files = list(train_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV found in /opt/ml/input/data/train")

    csv_path = csv_files[0]
    logger.info(f"Loading training data from: {csv_path}")

    df = pd.read_csv(csv_path)

    logger.info(f"Loaded rows: {len(df)}, columns: {len(df.columns)}")
    logger.debug(f"Feature Schema recieved: {list(df.columns)}")

    # ---- Convert to numpy ----
    np_array = df.to_numpy(dtype=np.float32)

    # ---- Log numpy details ----
    logger.info(f"Numpy array created | shape={np_array.shape} | dtype={np_array.dtype}")
    logger.debug(f"Numpy preview (first 3 rows):\n{np_array[:3]}")

    return np_array

# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------

def main():

    logger.info("========== SageMaker Training Job Started ==========")

    # ================================================================
    # Load Data
    # ================================================================
    raw_data = load_training_csv()

    # ================================================================
    # M1 — PREPROCESSING
    # ================================================================
    logger.info("M1: Preprocessing")

    sequences = preprocessing.to_sequences(raw_data, DEFAULT_SEQUENCE_LENGTH)

    logger.info(
        "Prepared sequences | samples=%d | seq_len=%d | features=%d",
        sequences.shape[0],
        sequences.shape[1],
        sequences.shape[2],
    )

    # ================================================================
    # M2 — MODEL INITIALIZATION
    # ================================================================
    logger.info("M2: Model Initialization")

    # dim_weights = preprocessing.load_dim_weights()
    model = model_init.initialize_model(
        sequence_length=sequences.shape[1],
        num_features=sequences.shape[2],
        dim_weights=DEFAULT_DIM_WEIGHTS,
    )

    # ================================================================
    # M3 — TRAINING
    # ================================================================
    logger.info("M3: Training")

    model_path = trainer.train_model(
        model,
        sequences,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    logger.info("Model trained and saved: %s", model_path)

    # ================================================================
    # M4 — CALIBRATION
    # ================================================================
    logger.info("M4: Calibration")

    metadata_path = calibration.calibrate(
        model_path, 
        sequences, 
        sequence_length=sequences.shape[1],
        num_features=sequences.shape[2],
        dim_weights=DEFAULT_DIM_WEIGHTS
    )

    logger.info("Calibration complete: %s", metadata_path)

    # ================================================================
    # M6 — EXPORT
    # ================================================================
    logger.info("M6: ONNX Export")

    onnx_path = export_onnx.export_model(
        model_path, 
        sequence_length=sequences.shape[1],
        num_features=sequences.shape[2],
        dim_weights=DEFAULT_DIM_WEIGHTS
    )

    logger.info("ONNX model exported: %s", onnx_path)

    logger.info("========== Training Pipeline Completed Successfully ==========")


if __name__ == "__main__":
    main()
