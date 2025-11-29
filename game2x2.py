"""Class for handling 2x2 SLOB Game."""

import numpy as np

from bidding import bidding_outcomes
from state import State
from transitions import apply_cell, check_winner, legal_moves
from utilities import weighted_random_choice


class SLOBGame2x2:
    """Class representing the 2x2 SLOB Game."""

    def __init__(self, board_probs):
        """Initialize the game with board probabilities.

        Args:
            board_probs (list): A list of probabilities for each cell on the board.

        """
        self.board_probs = board_probs
        self.win_lines = [
            (0, 1),
            (2, 3),
            (0, 2),
            (1, 3),
        ]

    def is_terminal(self, board: tuple) -> tuple[bool, None | float]:
        """Check if the game has reached a terminal state.

        Args:
            board (tuple): The current state of the board.

        Returns:
            tuple: A boolean indicating if the game is over and the winner (0, 1, or 0.5
            for draw).

        """
        # Check if there is a winner
        is_won, outcome = check_winner(self.win_lines, board)
        if is_won:
            return True, outcome

        # Full board -> draw
        if all(x is not None for x in board):
            return True, 0.5

        return False, None

    def successors(self, state: State) -> dict[State, float]:
        """Generate all possible successor states from the current state.

        Args:
            state (State): The current game state.

        Returns:
            dict: A dictionary mapping next states to their probabilities.

        """
        term, _ = self.is_terminal(state.board)
        if term:
            return {}  # No successors

        succ: dict[State, float] = {}  # map: State -> probability

        # Enumerate all bidding outcomes
        for bx, bo, winner, nx, no, next_adv in bidding_outcomes(state):
            Bx, Bo = state.x_chips, state.o_chips  # noqa: F841
            # Probability of choosing (bx,bo) under UNINFORMED random bidding:
            # DP won't use this; simulation might.
            # For DP, this isn't needed; bidding is a minimax choice.
            # But we keep computation structure simple.

            # Winner chooses a move; in DP, both players choose BEST move
            moves = legal_moves(state.board)

            for cell in moves:
                # Let’s NOT assume greedy; just enumerate all moves.
                # DP will select via argmax/argmin later.
                for next_board, p_outcome in apply_cell(
                    self.board_probs, state.board, cell, winner
                ):
                    next_state = State(
                        board=next_board, x_chips=nx, o_chips=no, advantage=next_adv
                    )

                    succ[next_state] = succ.get(next_state, 0.0) + p_outcome

        total = sum(succ.values())
        for k in succ:
            succ[k] /= total

        return succ

    def take_turn(
        self, state: State, bid_policy="random", move_policy="greedy"
    ) -> State:
        """Simulate a single turn in the game.

        Args:
            state (State): The current game state.
            bid_policy (str): The bidding policy to use (default is "random").
            move_policy (str): The move selection policy to use (default is "greedy").

        Returns:
            State: The next game state after the turn.

        """
        Bx, Bo = state.x_chips, state.o_chips
        adv = state.advantage
        board = state.board

        # ----------------------
        # 1. Choose bids (simulation only)
        # ----------------------
        if bid_policy == "random":
            bx = np.random.randint(Bx + 1) if Bx > 0 else 0
            bo = np.random.randint(Bo + 1) if Bo > 0 else 0
        else:
            raise NotImplementedError("Only random bidding for now.")

        # Determine winner
        if bx > bo:
            winner = 0
            next_x = Bx - bx
            next_o = Bo + bx
        elif bo > bx:
            winner = 1
            next_x = Bx + bo
            next_o = Bo - bo
        else:
            winner = adv
            if adv == 0:
                next_x = Bx - bx
                next_o = Bo + bx
            else:
                next_x = Bx + bo
                next_o = Bo - bo
            adv = 1 - adv  # flip advantage on tie

        # ----------------------
        # 2. Select move
        # ----------------------
        moves = legal_moves(board)

        if move_policy == "greedy":
            # The same rule you used before:
            # maximize (p_plus - p_minus)
            scores = self.board_probs[:, 0] - self.board_probs[:, 2]
            # sorted descending, but filter to legal moves
            best_move = max(moves, key=lambda c: scores[c])
            chosen_cell = best_move
        elif move_policy == "random":
            chosen_cell = np.random.choice(moves)
        else:
            raise NotImplementedError("Only greedy and random policies supported.")

        # ----------------------
        # 3. Sample outcome of placing mark
        # ----------------------
        outcomes = apply_cell(self.board_probs, board, chosen_cell, winner)
        # outcomes is list of (next_board, p)
        next_board, _ = weighted_random_choice(outcomes)

        # ----------------------
        # 4. Return next state
        # ----------------------
        return State(next_board, next_x, next_o, adv)
