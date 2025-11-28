"""Tests for bidding.py."""

from bidding import bidding_outcomes
from constants import P1
from state import State


def test_bidding():
    """Test for bidding_outcomes."""
    s = State(board=(None,) * 4, x_chips=2, o_chips=3, advantage=P1)
    outs = bidding_outcomes(s)

    # Check number of outcomes
    assert len(outs) == (2 + 1) * (3 + 1)

    # Check a specific case:
    # X bids 2, O bids 1 → X wins, pays 2 chips
    bx, bo, winner, nx, no, adv = [o for o in outs if o[:2] == (2, 1)][0]
    assert winner == P1
    assert nx == 0 and no == 5
