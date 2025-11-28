"""Utilities for SLOB games."""

import numpy as np


def weighted_random_choice(outcomes):
    """Select a random outcome based on given probabilities.

    Args:
        outcomes (list): A list of tuples where each tuple contains a board state and
        its probability.

    Returns:
        tuple: A randomly selected board state and its probability.

    """
    boards, probs = zip(*outcomes)
    idx = np.random.choice(len(boards), p=probs)
    return boards[idx], probs[idx]
