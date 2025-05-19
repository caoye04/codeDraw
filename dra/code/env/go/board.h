#ifndef _GOGAME
#define _GOGAME

#include <iostream>
#include <cstdlib>
#include <vector>

const int MAX_N = 21;

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

    std::vector<T> get_data_copy()const{
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

class c_GoBoard {
public:
    int n;

private:

    array2d<int> board;
    
    coord<int> ko_pos;
    
    int consecutive_pass;
    
    int step_cnt;

    coord<int> last_step_pos;
    int last_step_color;

    int count_stone(const int &color);

    bool remove_dead(const int & color, bool do_remove = true);

    bool ok_to_add_stone(const coord<int> &pos, const int &color);

public:
    c_GoBoard(const int &n);

    bool add_stone(const coord<int> &pos, const int &color); // add a stone at pos, return false if illegal
    void pass_stone(const int &color); // take an action that no stone is added
    const std::vector <int> & get_board() const; // get the current board as an array
    void set_board(const std::vector <int> &board_data); // set the current board by an array
    const std::vector <coord<int> > get_legal_moves(const int &color); // get all legal moves for a specific color
    int is_game_over(); // check if the game is over, and return the winner if it is
    coord<float> get_score(); // get the score of the game
    
    // the following method are for mannual features of board
    std::vector <int> get_liberty_map(); // get an array that represent the liberty of each position 
    std::vector <int> count_neighbor_map(const int &color);

};

#endif