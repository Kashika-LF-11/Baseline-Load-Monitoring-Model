"""
export_onnx.py
--------------
Exports trained Keras autoencoder into ONNX format.

This module intentionally:
• Rebuilds the model architecture
• Loads saved weights (.keras)
• Converts using tf2onnx
• Saves model.onnx into artifacts directory

No training state is reused — export is reproducible.
"""

from pathlib import Path
import tensorflow as tf
import tf2onnx

from config import logging_config, paths
from config.constants import DEFAULT_SEQUENCE_LENGTH, DEFAULT_DIM_WEIGHTS
from modules.model_init import initialize_model

logger = logging_config.get_logger(__name__)


# ---------------------------------------------------------------------
# Main Export Function
# ---------------------------------------------------------------------
def export_model(
    keras_model_path: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    num_features: int = None,
    dim_weights: list[float] = DEFAULT_DIM_WEIGHTS,
) -> Path:
    """
    Convert trained Keras model to ONNX format.

    Returns
    -------
    Path to model.onnx
    """

    logger.info("START MODULE: export_onnx")

    onnx_model_dir = paths.model_dir()
    onnx_output_path = onnx_model_dir / "model.onnx"

    # -----------------------------------------------------------------
    # Validate trained model exists
    # -----------------------------------------------------------------
    if not keras_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {keras_model_path}. "
            "Training likely failed before export."
        )

    logger.info("Found trained model: %s", keras_model_path)

    
    logger.info(
        "Rebuilding architecture | seq_len=%d | features=%d",
        sequence_length,
        num_features,
    )

    # -----------------------------------------------------------------
    # Recreate model & load weights
    # -----------------------------------------------------------------
    model = initialize_model(
        sequence_length=sequence_length,
        num_features=num_features,
        dim_weights=dim_weights,
    )

    logger.info("Loading weights into model")
    model.load_weights(keras_model_path)

    # -----------------------------------------------------------------
    # Define ONNX input signature
    # Explicit shape prevents runtime mismatch issues
    # -----------------------------------------------------------------
    spec = (
        tf.TensorSpec(
            shape=(None, sequence_length, num_features),
            dtype=tf.float32,
            name="input_sequence",
        ),
    )

    logger.info("Converting to ONNX (opset=15)")

    # -----------------------------------------------------------------
    # Get the actual model from the model wrapper: 
    # This is necessary because tf2onnx expects a tf.keras.Model, not a custom wrapper class
    # -----------------------------------------------------------------
    actual_model = model.model    

    # -----------------------------------------------------------------
    # Convert
    # -----------------------------------------------------------------
    onnx_model, _ = tf2onnx.convert.from_keras(
        actual_model,
        input_signature=spec,
        opset=15,
        output_path=str(onnx_output_path),
    )

    logger.info("ONNX export successful: %s", onnx_output_path)
    logger.info("END MODULE: export_onnx")

    return onnx_output_path
