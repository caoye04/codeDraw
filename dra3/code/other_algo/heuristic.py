from env import GoGame
def go_evaluate_position(board, x, y):
        """
        根据棋子的位置计算棋子的分数。
        """
        N = board.shape[0]
        score = 0
        stone = board[x][y]
        if x == 0 or x == N-1 or y == 0 or y == N-1:
            score = 0
        elif N/3 <= x <= N/3*2 and N/3 <= y <= N/3*2:
            score += .2
        else:
            score += .5
        # 检查周围n*n格内没有其他棋子的情况
        for n in range(1, 5):  # 检查1到4格的范围
            for dx in range(-n, n+1):
                for dy in range(-n, n+1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == 0:
                        break
                else:  # 如果没有找到其他棋子，说明当前n成立
                    score += 0.05 * n
                    break
            else:
                continue
            break

        # 检查周围8格是否连上自己方棋子
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == stone:
                score += 0.1  # 8格连接加分

        # 检查周围4格是否连上自己方棋子
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == stone:
                score += 0.1  # 4格连接加分

        # 检查周围4格是否有三个及以上我方棋子
        same_color_count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and board[nx][ny] == stone:
                same_color_count += 1
        if same_color_count >= 3:
            score -= 0.3

        return score

def go_heuristic_evaluation(game_state: GoGame):
    """
    启发式评估函数，计算整个棋盘的总分。
    """
    total_score = 0
    board = game_state.board_array
    N = board.shape[0]
    current_player = game_state.current_player
    stone_number = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] != 0:
                if board[i][j] == current_player:
                    total_score += go_evaluate_position(board, i, j)
                    stone_number += 1
    return total_score / (stone_number + 1)