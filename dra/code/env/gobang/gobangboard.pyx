# distutils: language=c++
from gobangboard cimport c_GobangBoard, coord
from env.base_env import DRAW

import numpy as np
from typing import List, Tuple

cdef class GobangBoard():
    cdef c_GobangBoard*  c_GobangBoard_ptr
    def __init__(self, h:int, w:int, n:int) -> None:
        self.c_GobangBoard_ptr = new c_GobangBoard(h, w, n)
    
    def __dealloc__(self):
        del self.c_GobangBoard_ptr
    
    def get_legal_moves(self) -> List[Tuple(int, int)]:
        cdef vector[coord] l = self.c_GobangBoard_ptr.get_legal_moves()
        return [(c.x, c.y) for c in l]
    
    def to_numpy(self, dtype=np.int32) -> np.ndarray:
        r = np.asarray(self.c_GobangBoard_ptr.get_board(), dtype=dtype)
        r.resize(self.c_GobangBoard_ptr.h, self.c_GobangBoard_ptr.w)
        return r

    def check_winner(self) -> int:
        return self.c_GobangBoard_ptr.check_winner()
    
    def execute_move(self, move:Tuple[int, int], color:int) -> None:
        assert move[0] >= 0 and move[0] < self.c_GobangBoard_ptr.h, f"invalid move[0]: {move[0]}, h: {self.c_GobangBoard_ptr.h}"
        assert move[1] >= 0 and move[1] < self.c_GobangBoard_ptr.w, f"invalid move[1]: {move[1]}, w: {self.c_GobangBoard_ptr.w}"
        x, y = move
        self.c_GobangBoard_ptr.execute_move(coord(x, y), color)
    
    def has_legal_moves(self) -> bool:
        return self.c_GobangBoard_ptr.has_legal_moves()

    def reset(self) -> 'GobangBoard':
        self.c_GobangBoard_ptr.reset()
        return self

    def copy(self):
        cdef GobangBoard new_board = GobangBoard(
            self.c_GobangBoard_ptr.h, self.c_GobangBoard_ptr.w, self.c_GobangBoard_ptr.n
        )
        new_board.c_GobangBoard_ptr[0] = self.c_GobangBoard_ptr[0]
        return new_board