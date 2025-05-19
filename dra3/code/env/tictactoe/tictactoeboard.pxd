# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "board.cpp":
    pass

cdef extern from "board.h":
    cdef cppclass coord[int]:
        int x, y
        coord() except +
        coord(int, int) except +
    
    cdef cppclass c_TicTacToeBoard:
        int h, w, n
        c_TicTacToeBoard(int, int, int) except +
        vector[coord[int]] get_legal_moves() except +
        int check_winner() except +
        void execute_move(coord[int], int) except +
        bool has_legal_moves() except +
        void reset() except +
        vector[int] get_board() except +