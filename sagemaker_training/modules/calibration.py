"""
calibration.py
--------------
Computes reconstruction error statistics used for anomaly detection.

This module EXACTLY mirrors the notebook calibration logic.

Steps:
1) Recreate model architecture
2) Load trained weights (.keras)
3) Run reconstruction on training data
4) Compute RMSE error distribution for target signal
5) Save limits into metadata.json
"""

from pathlib import Path
import json
import numpy as np

from config import logging_config, paths
from modules.model_init import initialize_model

logger = logging_config.get_logger(__name__)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def _compute_rmse_stats(
        x_true: np.ndarray, 
        x_pred: np.ndarray, 
        timestep: int, 
        dim: int):
    """
    Replicates notebook get_mean_and_std()
    """

    error = x_true[:, timestep, dim] - x_pred[:, timestep, dim]

    mean_rmse = float(np.sqrt(np.mean(np.square(error))))
    std_rmse = float(np.std(error))

    upper_limit = float(mean_rmse + 2 * std_rmse)
    lower_limit = float(mean_rmse - 2 * std_rmse)

    return {
        "mean_error": mean_rmse,
        "std_error": std_rmse,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
    }


# ---------------------------------------------------------------------
# Main Calibration
# ---------------------------------------------------------------------

def calibrate(
        model_path: Path,
        x_train: np.ndarray, 
        sequence_length: int, 
        num_features: int, 
        dim_weights: list[float]
    ) -> Path:
    """
    Perform calibration using trained autoencoder.

    Returns
    -------
    Path to metadata.json
    """

    logger.info("START MODULE: calibration")

    # Define target to assess errors for: 
    target_timestep = 4   # this is the centre datapoint in the 10-timestep sequence
    target_dimension = 0  # this represents the load feature within the feature sequence to consider

    logger.info(
        "Calibration target | timestep=%d | dimension=%d",
        target_timestep,
        target_dimension,
    )

    # -----------------------------------------------------------------
    # Recreate model & load weights
    # -----------------------------------------------------------------
    model = initialize_model(sequence_length, num_features, dim_weights)

    # Check if model path exists before loading weights from it: 
    if not model_path.exists():
        raise FileNotFoundError(
            "model.keras not found — training likely failed"
        )
    
    logger.info("Loading trained weights: %s", model_path)
    model.load_weights(model_path)

    # -----------------------------------------------------------------
    # Reconstruction prediction
    # -----------------------------------------------------------------
    logger.info("Running reconstruction inference for calibration")
    predictions = model.predict(x_train)

    # -----------------------------------------------------------------
    # Compute limits (Notebook logic)
    # -----------------------------------------------------------------
    metadata = _compute_rmse_stats(
        x_train,
        predictions,
        target_timestep,
        target_dimension,
    )

    # -----------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------
    metadata_path = paths.artifacts_dir() / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Calibration results: %s", metadata)
    logger.info("END MODULE: calibration | saved=%s", metadata_path)

    return metadata_path
