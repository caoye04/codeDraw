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

    cdef cppclass c_GoBoard:
        int n
        int* _data_buffer
        c_GoBoard() except +
        c_GoBoard(int) except +
        vector[int] get_board() except +
        vector[int] get_liberty_map() except +
        vector[int] count_neighbor_map(int) except +
        bool add_stone(coord[int], int) except +
        void pass_stone(int) except +
        vector[coord[int]] get_legal_moves(int) except +
        coord[float] get_score() except +
        void set_board(vector[int]) except +
        int is_game_over() except +