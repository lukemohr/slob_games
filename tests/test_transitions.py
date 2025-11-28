"""Tests for transitions.py."""

import numpy as np

from constants import P1
from transitions import apply_cell


def test_apply_cell():
    """Test apply_cell."""
    board = (None, None, None, None)
    probs = np.array(
        [
            [1.0, 0.0, 0.0],  # always good
            [0.0, 1.0, 0.0],  # always neutral
            [0.0, 0.0, 1.0],  # always bad
            [0.5, 0.5, 0.0],
        ]
    )  # mix

    # always good
    out = apply_cell(probs, board, 0, P1)
    assert out == [(((P1, None, None, None)), 1.0)]
