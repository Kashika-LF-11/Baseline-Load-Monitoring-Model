"""
Preprocessing utilities.

PURE FUNCTIONS ONLY.
No disk I/O.
No logging side effects.
"""

import numpy as np
from config import paths, logging_config

logger = logging_config.get_logger(__name__)




def to_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Convert 2D array -> 3D sliding windows

    Input:
        (time_steps, features)

    Output:
        (samples, sequence_length, features)
    """

    n_samples = len(data) - sequence_length
    if n_samples <= 0:
        raise ValueError(
            f"Sequence length {sequence_length} is larger than data length {len(data)}"
        )

    sequences = []
    for i in range(n_samples):
        sequences.append(data[i : i + sequence_length])

    return np.array(sequences)
