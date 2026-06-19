from pathlib import Path
import os


def env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default))

# place to fetch the training data from 
def input_data_dir() -> Path:
    return env_path("SM_INPUT_DATA_DIR", "/opt/ml/input/data/train")

# where the final onnx model will be saved 
def model_dir() -> Path:
    return env_path("SM_MODEL_DIR", "/opt/ml/model")

# For intermediate model.kerads and metadata.json files
def intermediate_dir() -> Path:
    return env_path("SM_INTERMEDIATE_DIR", "/opt/ml/output/intermediate")

# For metadata.json 
def artifacts_dir() -> Path:
    return intermediate_dir() / "artifacts"

# For model.keras
def keras_model_path() -> Path:
    return artifacts_dir() / "models"