"""Functions for simulating games."""

from game2x2 import SLOBGame2x2
from state import State


def simulate_game(state: State, game: SLOBGame2x2) -> float:
    """Simulate a single turn in the game.

    Args:
        state (State): The current game state.
        game (SLOBGame2x2): The game to simulate.

    Returns:
        float: The winner of the game (1 for player 1, 0 for player 2, 0.5 for draw)

    """
    while True:
        term, winner = game.is_terminal(state.board)
        if term:
            return winner
        state = game.take_turn(state)
