"""
trainer.py
----------
Implements the EXACT training behavior from the research notebook.

This module does NOT modify training logic.
It only adapts notebook training to SageMaker filesystem paths.

Contract:
- Receives initialized LSTMTimeSeriesAutoencoder
- Runs reconstruction training
- Saves BEST model weights (.keras)
- No optimization tricks (no early stopping, schedulers, etc.)

This ensures calibration statistics remain consistent with offline experiments.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

from config import logging_config, paths
from config.constants import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
from modules.model_init import LSTMTimeSeriesAutoencoder

logger = logging_config.get_logger(__name__)


# ---------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------

def train_model(
    model: LSTMTimeSeriesAutoencoder,
    x_train: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """
    Train the autoencoder exactly like the notebook.

    Parameters
    ----------
    model : LSTMTimeSeriesAutoencoder
        Initialized model from model_init
    x_train : np.ndarray
        Shape (samples, seq_len, features)

    Returns
    -------
    Path to best saved .keras model
    """

    logger.info("START MODULE: trainer")
    logger.info("Training samples: %d", x_train.shape[0])
    logger.info("Sequence length: %d | Features: %d", x_train.shape[1], x_train.shape[2])

    # -----------------------------------------------------------------
    # Define deterministic SageMaker model path
    # -----------------------------------------------------------------
    model_dir = paths.keras_model_path()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.keras"

    # -----------------------------------------------------------------
    # Create checkpoint callback 
    # -----------------------------------------------------------------
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min'
        )

    # -----------------------------------------------------------------
    # Train (STRICTLY matches notebook arguments)
    # -----------------------------------------------------------------
    logger.info(
        "Training autoencoder | epochs=%d | batch_size=%d | shuffle=False",
        epochs,
        batch_size,
    )

    model.compile_model()
    model.train(
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
    )

    # -----------------------------------------------------------------
    # Verify model exists
    # -----------------------------------------------------------------
    if not model_path.exists():
        raise RuntimeError(
            "Training completed but model.keras not created. Check training logs."
        )

    logger.info("END MODULE: trainer | best_model=%s", model_path)

    return model_path
