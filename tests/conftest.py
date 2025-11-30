"""Test config."""

import numpy as np
import pytest

from game2x2 import SLOBGame2x2


@pytest.fixture
def game2x2():
    """Create simple 2x2 game."""
    # Deterministic board_probs for simplified tests
    probs = np.array(
        [
            [1.0, 0.0, 0.0],  # always + for cell 0
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return SLOBGame2x2(board_probs=probs, total_chips=3)
