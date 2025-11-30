"""Run simulations."""

import numpy as np

from game2x2 import SLOBGame2x2
from simulation import simulate_game
from state import State


def main():
    """Run the simulation."""
    num_games = 10000
    wins = {"X": 0, "O": 0, "draw": 0}

    for _ in range(num_games):
        board = (None, None, None, None)
        board_probs = np.array(
            [[0.2, 0.4, 0.4], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.5, 0.25, 0.25]]
        )
        x_chips = 3
        o_chips = 3
        # X=0, O=1
        advantage = 0
        state = State(board, x_chips, o_chips, advantage)
        game = SLOBGame2x2(board_probs, x_chips + o_chips)
        winner = simulate_game(state, game)
        if winner == 0:
            wins["X"] += 1
        elif winner == 1:
            wins["O"] += 1
        else:
            wins["draw"] += 1
    print(wins)


if __name__ == "__main__":
    main()
