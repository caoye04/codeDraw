from ..base_env import EMPTY
import numpy as np

class TicTacToeBoard:
    def __init__(self, h=3, w=3, n=3):
        self.w = w
        self.h = h
        self.n = n
        self.pieces = np.zeros((self.h, self.w), dtype=int)

    def get_legal_moves(self):
        moves = []
        for x in range(self.h):
            for y in range(self.w):
                if self.pieces[x][y] == EMPTY:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self):
        return (self.pieces == 0).any()

    def check_win_status(self, player: int):
        for h in range(self.h):
            for w in range(self.w):
                if self.pieces[h][w] != player:
                    continue
                if h in range(self.h - self.n + 1) and len(set(self.pieces[i][w] for i in range(h, h + self.n))) == 1:
                    return True
                if w in range(self.w - self.n + 1) and len(set(self.pieces[h][j] for j in range(w, w + self.n))) == 1:
                    return True
                if h in range(self.h - self.n + 1) and w in range(self.w - self.n + 1) and len(set(self.pieces[h + k][w + k] for k in range(self.n))) == 1:
                    return True
                if h in range(self.h - self.n + 1) and w in range(self.n - 1, self.w) and len(set(self.pieces[h + l][w - l] for l in range(self.n))) == 1:
                    return True
        return False

    def execute_move(self, move, player):
        x, y = move
        assert self.pieces[x][y] == EMPTY, f"Invalid move: {move} for player {player} : place is already occupied."
        self.pieces[x][y] = player