#include "board.h"

#include <cstring>
#include <queue>

const int INF = 1e9;
const int WHITE = -1, BLACK = 1, EMPTY = 0, DRAW = 2;

const int dx[4] = {0, 0, 1, -1};
const int dy[4] = {1, -1, 0, 0};


c_GoBoard::c_GoBoard(const int &n):
    n(n),
    board(n, n),
    ko_pos(-INF, -INF),
    consecutive_pass(0),
    step_cnt(0),
    last_step_pos(-INF, -INF),
    last_step_color(-INF)
{
    if (n > MAX_N){
        std::cerr << "Board size(" << n << ") too large! (max:" << MAX_N << ") " << std::endl;
        throw std::exception();
    }
}

const std::vector <int> & c_GoBoard::get_board() const{
    return board.get_data();
}

void c_GoBoard::set_board(const std::vector <int> &board_data){
    if (board_data.size() != (unsigned int) n * n){
        std::cerr << "[c_GoBoard::set_board] invalid board data size: " << board_data.size() << " (expect: " << n * n << ")" << std::endl;
        throw std::exception();
    }
    std::memcpy(board.get_data_ptr(), board_data.data(), sizeof(int) * n * n);
}

int c_GoBoard::count_stone(const int &color){
    int cnt = 0;
    for(int i=0; i<this->n; i++)
        for(int j=0; j<this->n; j++)
            if(this->board[i][j] == color)
                cnt++;
    return cnt;
}

bool c_GoBoard::remove_dead(const int & color, bool do_remove){
    // Remove dead stones of a specific color
    // @param color: color of the stones
    // @param do_remove: whether to remove the dead stones
    // @return: whether the player's own stone is removed
    // NOTE: this function will also update ko_pos when do_remove is true
    //       it is not only used to update the board, but also to check if a move is legal
    auto visited = array2d<int>(n, n, false), removed = array2d<int>(n, n, false);
    coord<int> first_capture_pos = coord<int>(-INF, -INF);
    bool own_stone_removed = false;
    int total_stone_removed = 0;

    for (int c : {-color, color}) {
        for (int x = 0; x < n ; x++) {
            for (int y = 0; y < n; y++) {
                if(board[x][y] == c && !visited[x][y]){
                    int liberties = 0;
                    std::queue<coord<int>> q, group;
                    q.push(coord<int>(x, y));
                    group.push(coord<int>(x, y));
                    visited[x][y] = true;
                    while (!q.empty()){
                        coord<int> p = q.front();
                        q.pop();
                        int px = p.x, py = p.y;
                        for (int t = 0; t < 4; t++){
                            int nx = px + dx[t], ny = py + dy[t];
                            if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                            if (board[nx][ny] == EMPTY || (board[nx][ny] == -c && removed[nx][ny])){
                                liberties++;
                            } 
                            else if (board[nx][ny] == c && !visited[nx][ny]){
                                q.push(coord<int>(nx, ny));
                                group.push(coord<int>(nx, ny));
                                visited[nx][ny] = true;
                            }
                        }
                    }
                    if (liberties == 0) {
                        if (first_capture_pos.x == -INF){
                            first_capture_pos = group.front();
                        }
                        total_stone_removed += group.size();
                        own_stone_removed  |= (c == color);
                        while (!group.empty()){
                            coord<int> p = group.front();
                            group.pop();
                            removed[p.x][p.y] = true;
                            if (do_remove) board[p.x][p.y] = EMPTY;
                        }
                    }
                }
            }
        }
    }
    if (do_remove)
        ko_pos = total_stone_removed == 1 ? first_capture_pos : coord<int>(-INF, -INF);
    return own_stone_removed;
}

void c_GoBoard::pass_stone(const int &color){
    step_cnt += 1;
    consecutive_pass += 1;
    ko_pos = coord<int>(-INF, -INF);
}

bool c_GoBoard::ok_to_add_stone(const coord<int> &pos, const int &color){
    // Check if it is legal to add a stone at pos
    // @param pos: position to add stone
    // @param color: color of the stone
    // @return: whether it is legal to add a stone
    const int &x = pos.x, &y = pos.y;
    if (x < 0 || x >= n || y < 0 || y >= n || board[x][y] != EMPTY)
        return false;
    if (pos.x == ko_pos.x && pos.y == ko_pos.y)
        return false;
    for (int t = 0; t < 4; t++){
        int nx = x + dx[t], ny = y + dy[t];
        if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
        if (board[nx][ny] == EMPTY)
            return true;
    }
    // try if the move is legal
    board[x][y] = color;
    auto own_stone_removed = remove_dead(color, false);
    board[x][y] = EMPTY;
    return !own_stone_removed;
}

bool c_GoBoard::add_stone(const coord<int> &pos, const int &color){
    // Add a stone at pos, throw exception if illegal
    // @param pos: position to add stone
    // @param color: color of the stone
    // @return: always true 
    // NOTE: this function not only add a stone, but also remove dead stones
    const int &x = pos.x, &y = pos.y;
    if (x < 0 || x >= n || y < 0 || y >= n){
        std::cerr << "[c_GoBoard::add_stone] invalid pos (" << x << ", " << y << ")" << std::endl;
        throw std::exception();
    }
    if (board[x][y] != EMPTY){
        std::cerr << "[c_GoBoard::add_stone] position not empty: pos=(" << x << ", " << y << "), color here=" << board[x][y] << std::endl;
         throw std::exception();
    }
    board[x][y] = color;
    auto own_stone_removed = remove_dead(color, true);
    if (own_stone_removed){
        std::cerr << "[c_GoBoard::add_stone] stone has no liberty after placed: pos=(" << x << ", " << y << "), color=" << color << std::endl;
        throw std::exception();
    }
    consecutive_pass = 0;
    step_cnt += 1;
    last_step_pos = pos;
    last_step_color = color;
    return true;
}

const std::vector <coord<int> > c_GoBoard::get_legal_moves(const int &color){
    // Get all legal moves for a specific color
    // @param color: color of the stones
    // @return: a list of legal moves
    std::vector <coord<int> > legal_moves;
    for (int x = 0; x < n; x++){
        for (int y = 0; y < n; y++){
            if (board[x][y] == EMPTY && ok_to_add_stone(coord<int>(x, y), color)){
                legal_moves.push_back(coord<int>(x, y));
            }
        }
    }
    return legal_moves;
}

coord<float> c_GoBoard::get_score(){
    // Get the score of the game
    // @return: a pair of float, (black score, white score)
    // NOTE: the score is the number of stones and empty points captured by each player
    float scores[2] = {.0, .0}; // white, black
    auto visited = array2d<int>(n, n, false), removed = array2d<int>(n, n, false);
    for (int x = 0; x < n; x++){
        for (int y = 0; y < n; y++){
            if (board[x][y] == WHITE) {
                scores[0] += 1;
            }
            else if (board[x][y] == BLACK){
                scores[1] += 1;
            } 
            else if (!visited[x][y]){ // EMPTY
                int cnt = 1;
                visited[x][y] = true;
                int color = EMPTY;
                std::queue<coord<int>> q;
                q.push(coord<int>(x, y));
                while (!q.empty()){
                    coord<int> p = q.front();
                    q.pop();
                    int px = p.x, py = p.y;
                    for (int t = 0; t < 4; t++){
                        int nx = px + dx[t], ny = py + dy[t];
                        if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                        if (board[nx][ny] == EMPTY && !visited[nx][ny]){
                            q.push(coord<int>(nx, ny));
                            visited[nx][ny] = true;
                            cnt++;
                        }
                        else if (board[nx][ny] != EMPTY){
                            if (color == EMPTY){
                                color = board[nx][ny];
                            }
                            else if (color != board[nx][ny]){
                                color = -INF;
                                break;
                            }
                        }
                    }
                }
                if (color == WHITE){
                    scores[0] += cnt;
                }
                else if (color == BLACK){
                    scores[1] += cnt;
                }
            }
            
        }
    }
    return coord<float>(scores[1], scores[0]); // black, white
}

int c_GoBoard::is_game_over(){
    // Check if the game is over, and return the winner if it is
    // @return: the winner of the game (BLACK, WHITE or DRAW), or EMPTY if the game is not over
    if (step_cnt >= n * n * 2 + 1 || consecutive_pass >= 2){
        auto scores = get_score();
        if (scores.x > scores.y){
            return BLACK;
        }
        else if (scores.x < scores.y){
            return WHITE;
        }
        else{
            return DRAW;
        }
    }
    return EMPTY;
}

std::vector <int> c_GoBoard::get_liberty_map(){
    // Get the liberty count of the cluster of each postion belongs to
    // @return: a 1D vector of size (x*y), flatten from the liberty count map.
    auto visited = array2d<int>(n, n, false), liberty_map = array2d<int>(n, n, 0);
    for (int x = 0; x < n ; x++) {
        for (int y = 0; y < n; y++) {
            if(!visited[x][y] && board[x][y]){
                int c = board[x][y];
                int liberties = 0;
                std::queue<coord<int>> q, group;
                q.push(coord<int>(x, y));
                group.push(coord<int>(x, y));
                visited[x][y] = true;
                while (!q.empty()){
                    coord<int> p = q.front();
                    q.pop();
                    int px = p.x, py = p.y;
                    for (int t = 0; t < 4; t++){
                        int nx = px + dx[t], ny = py + dy[t];
                        if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                        if (board[nx][ny] == EMPTY ){
                            liberties++;
                        } 
                        else if (board[nx][ny] == c && !visited[nx][ny]){
                            q.push(coord<int>(nx, ny));
                            group.push(coord<int>(nx, ny));
                            visited[nx][ny] = true;
                        }
                    }
                }
                while (!group.empty()){
                    coord<int> p = group.front();
                    group.pop();
                    liberty_map[p.x][p.y] = liberties;
                }
            }
            else if (board[x][y]==0){
                liberty_map[x][y] = -1;
            }
        }
    }
    return liberty_map.get_data_copy();
}

std::vector <int> c_GoBoard::count_neighbor_map(const int &color){
    auto count_map = array2d<int>(n, n, 0);
    for (int x = 0; x < n ; x++) {
        for (int y = 0; y < n; y++) if (board[x][y]==color){
            for (int t = 0; t < 4; t++){
                int nx = x + dx[t], ny = y + dy[t];
                if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                count_map[nx][ny]+= 1;
            }
        }
    }
    return std::vector<int>(count_map.get_data_copy());
}