#ifndef TICTACTOE_BOARD
#define TICTACTOE_BOARD

#include <iostream>
#include <cstdlib>
#include <vector>

const int N = 3;

template <typename T>
struct array2d{
private:
    std::vector<T> data;
    int n, m;
public:
    array2d(int n, int m, int defaut_value = 0):n(n), m(m){
        data.resize(n*m, defaut_value);
    }

    int* operator[](int i){
        int* p = data.data();
        return p + i*m;
    }

    const std::vector<T> & get_data()const{
        return data;
    }

    T* get_data_ptr(){
        return data.data();
    }
};

template <typename T>
struct coord{
    T x, y;
    coord(): x(0), y(0){}
    coord(T x, T y): x(x), y(y){}
};

class c_GobangBoard {
public:
    int h, w;
    int n;
private:
    int step_cnt;
    array2d<int> board;
public:
    c_GobangBoard(int h, int w, int n):h(h), w(w), n(n), step_cnt(0), board(h, w){}
    const std::vector <coord<int> > get_legal_moves(); 
    int check_winner();
    void execute_move(const coord<int> &pos, const int &color);
    bool has_legal_moves();
    void reset();
    const std::vector <int> & get_board() const;
};

#endif