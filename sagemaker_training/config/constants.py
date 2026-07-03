"""Project constants and defaults."""
from typing import Final

DEFAULT_SEQUENCE_LENGTH: Final[int] = 10
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_EPOCHS: Final[int] = 50
DEFAULT_DIM_WEIGHTS: Final[list[float]] = [0.8, 0.05, 0.05, 0.05, 0.025]

PROGRAM_SCOPED_TOOL_ROTATION: Final[str] = "PROGRAM_SCOPED_TOOL_ROTATION"
CROSS_MACHINE_PROGRAM_TOOL_ROTATION: Final[str] = "CROSS_MACHINE_PROGRAM_TOOL_ROTATION"

WORKFLOW_DIM_WEIGHTS: Final[dict[str, list[float]]] = {
    PROGRAM_SCOPED_TOOL_ROTATION: [0.8, 0.05, 0.05, 0.05, 0.025],
    CROSS_MACHINE_PROGRAM_TOOL_ROTATION: [0.80, 0.05, 0.05, 0.05, 0.025, 0.025],
}


def resolve_dim_weights(tool_usage_key: str | None = None) -> list[float]:
    """Resolve the model training weights based on the incoming workflow key."""
    if tool_usage_key in WORKFLOW_DIM_WEIGHTS:
        return WORKFLOW_DIM_WEIGHTS[tool_usage_key]
    return list(DEFAULT_DIM_WEIGHTS)
