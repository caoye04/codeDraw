#include "board.h"

const int INF = 1e9;
const int WHITE = -1, BLACK = 1, EMPTY = 0, DRAW = 2, NOTEND = 0;

const int dx[4] = {0, 0, 1, -1};
const int dy[4] = {1, -1, 0, 0};

const std::vector <coord<int> > c_GobangBoard::get_legal_moves(){
    std::vector <coord<int> > legal_moves;
    for (int x = 0; x < h; x++){
        for (int y = 0; y < w; y++){
            if (board[x][y] == EMPTY){
                legal_moves.push_back(coord<int>(x, y));
            }
        }
    }
    return legal_moves;
}

int c_GobangBoard::check_winner(){
    // horizontal
    for (int x = 0; x < h; x++){
        for (int y = 0; y + n - 1 < w; y++) if (board[x][y] != EMPTY){
            bool ok = true;
            for (int i = 0; i < n; i++){
                if (board[x][y+i] != board[x][y]){
                    ok = false;
                    break;
                }
            }
            if (ok) return board[x][y];
        }
    }
    // vertical
    for (int y = 0; y < w; y++){
        for (int x = 0; x + n - 1 < h; x++) if (board[x][y] != EMPTY){
            bool ok = true;
            for (int i = 0; i < n; i++){
                if (board[x+i][y] != board[x][y]){
                    ok = false;
                    break;
                }
            }
            if (ok) return board[x][y];
        }
    }
    // diagonal
    for (int x = 0; x + n - 1 < h; x++){
        for (int y = 0; y + n - 1 < w; y++) if (board[x][y] != EMPTY){
            bool ok = true;
            for (int i = 0; i < n; i++){
                if (board[x+i][y+i] != board[x][y]){
                    ok = false;
                    break;
                }
            }
            if (ok) return board[x][y];
        }
    }
    // anti-diagonal
    for (int x = 0; x + n - 1 < h; x++){
        for (int y = n - 1; y < w; y++) if (board[x][y] != EMPTY){
            bool ok = true;
            for (int i = 0; i < n; i++){
                if (board[x+i][y-i] != board[x][y]){
                    ok = false;
                    break;
                }
            }
            if (ok) return board[x][y];
        }
    }
    return NOTEND;
}

void c_GobangBoard::execute_move(const coord<int> &pos, const int &color){
    board[pos.x][pos.y] = color;
    step_cnt += 1;
}

bool c_GobangBoard::has_legal_moves(){
    return step_cnt < h*w;
}

void c_GobangBoard::reset(){
    step_cnt = 0;
    for (int x = 0; x < h; x++){
        for (int y = 0; y < w; y++){
            board[x][y] = EMPTY;
        }
    }
}

const std::vector <int> & c_GobangBoard::get_board() const{
    return board.get_data();
}