# distutils: language=c++
from goboard cimport c_GoBoard, coord
import numpy as np
from env.base_env import DRAW


cdef class GoBoard():
    cdef c_GoBoard*  c_GoBoard_ptr
    def __init__(self, n:int):
        self.c_GoBoard_ptr = new c_GoBoard(n)

    def __dealloc__(self):
        del self.c_GoBoard_ptr

    @property
    def n(self):
        return self.c_GoBoard_ptr.n

    def to_numpy(self, dtype=np.int32):
        r = np.asarray(self.c_GoBoard_ptr.get_board(), dtype=dtype)
        r.resize(self.n, self.n)
        return r
    
    def count_neighbor_map(self, color:int, dtype=np.int32):
        r = np.asarray(self.c_GoBoard_ptr.count_neighbor_map(color), dtype=dtype)
        r.resize(self.n, self.n)
        return r

    def get_liberty_map(self, dtype=np.int32):
        r = np.asarray(self.c_GoBoard_ptr.get_liberty_map(), dtype=dtype)
        r.resize(self.n, self.n)
        return r
    
    def load_numpy(self, board:np.ndarray):
        board.resize(self.n*self.n)
        cdef vector[int] board_data = board.tolist()
        self.c_GoBoard_ptr.set_board(board_data)
    
    def copy(self):
        cdef GoBoard new_board = GoBoard(self.n)
        new_board.c_GoBoard_ptr[0] = self.c_GoBoard_ptr[0]
        return new_board
    
    def get_legal_moves(self, color:int):
        cdef vector[coord] l = self.c_GoBoard_ptr.get_legal_moves(color)
        return [(i.x, i.y) for i in l]
    
    def add_stone(self, x:int, y:int, color:int):
        self.c_GoBoard_ptr.add_stone(coord[int](x, y), color)
    
    def pass_stone(self, color:int):
        self.c_GoBoard_ptr.pass_stone(color)
    
    def get_score(self):
        cdef coord[float] r = self.c_GoBoard_ptr.get_score()
        return [r.x, r.y]
    
    def get_winner(self):
        r = self.c_GoBoard_ptr.is_game_over()
        if r == 2:
            return DRAW
        return r
    