from typing import List, Tuple
import numpy as np

WHITE, BLACK, EMPTY = -1, 1, 0

EPS = 1e-7

NOTEND = 0

DRAW = EPS

MAX_N = 21

class BaseGame:
    def __init__(self, n:int, m:int) -> None:
        self.n = n
        self.m = m
        self.board = None
        self._action_size = -1
        self._current_player = BLACK
        self._ended = False
    
    def _check_reset(self):
        assert not self._ended, "The game has ended! reset before doing anything."
        assert self.board is not None, "You should reset the game first!"
    
    def _copy_basic_info(self, game:'BaseGame'):
        self.n = game.n
        self.m = game.m
        self._action_size = game._action_size
        self._current_player = game._current_player
        self._ended = game._ended
    
    @property
    def current_player(self):
        return self._current_player
    
    @property
    def observation_size(self):
        return self.n, self.n
    
    @property
    def action_space_size(self):
        return self._action_size
    
    @property
    def action_mask(self):
        raise NotImplementedError
    
    @property
    def observation(self):
        return self.board
    
    @property
    def ended(self):
        return self._ended
    
    def fork(self) -> 'BaseGame':
        raise NotImplementedError
        
    def reset(self) -> np.ndarray:
        self._ended = False
        return None
    
    def step(self, action:int ) -> Tuple[np.ndarray, float, bool]:
        raise NotImplementedError
    
    def get_symmetries(self, policy:np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError
    
    def get_canonical_form_obs(self) ->np.ndarray:
        raise NotImplementedError
    
    def compute_canonical_form_obs(self, obs:np.ndarray, player:int)-> np.ndarray:
        raise NotImplementedError
    
    def to_string(self) -> str:
        return NotImplementedError
    
    def init_param_list(self) -> List:
        return NotImplementedError

def get_symmetries_single_square(board:np.ndarray, policy:np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    assert board.shape[0] == board.shape[1], f"board shape: {board.shape}"
    n = board.shape[0]
    is_go_game =  policy.shape[0] == n * n + 1
    if is_go_game:
        pi_board = np.reshape(policy[:-1], (n, n))
    else:
        pi_board = np.reshape(policy, (n, n))
    l = []

    for i in range(0, 5):
        for j in [True, False]:
            newB = np.rot90(board, i)
            newPi = np.rot90(pi_board, i)
            if j:
                newB = np.fliplr(newB)
                newPi = np.fliplr(newPi)
            if is_go_game:
                l += [(newB, list(newPi.ravel()) + [policy[-1]])]
            else:
                l += [(newB, list(newPi.ravel()) )]
    return l

def get_symmetries(board:np.ndarray, policy:np.ndarray)-> List[Tuple[np.ndarray, np.ndarray]]:
    if len(board.shape)==2 and board.shape[0] == board.shape[1]:
        return get_symmetries_single_square(board, policy)
    elif len(board.shape)==1:
        act_shape = len(policy)
        board_shape = (len(board)-1-act_shape)//4
        n = round(board_shape**0.5)
        assert board_shape*4+1+act_shape == board.shape[0]
        s = [board[i*board_shape: (i+1)*board_shape].reshape(n, n) for i in range(4)]
        v = board[4*board_shape: 4*board_shape+1]
        a = board[-act_shape:]
        result = []
        policy_symmetries = [i[-1] for i in get_symmetries_single_square(s[0], policy)]
        for i in range(4):
            l = get_symmetries_single_square(s[i], a)
            if not result:
                result = [(i[0].reshape(-1), i[1]) for i in l]
            else:
                result = [(np.concatenate([i[0], j[0].reshape(-1)], axis=0), i[1]) for i, j in zip(result, l)]
        result = [(np.concatenate([i[0], v, i[1]], axis=0), j) for i, j in zip(result, policy_symmetries)]
        # print("[DEBUG] result[0][0].shape:", result[0][0].shape)
        return result
            

class ResultCounter():
    def __init__(self, win=0, draw=0, lose=0):
        self.win = win
        self.draw = draw
        self.lose = lose
    
    def reset(self):
        self.win, self.draw, self.lose = 0, 0, 0
    
    def add(self, reward:float, last_player:int=1):
        if abs(reward) != DRAW:
            if reward * last_player > 0:
                self.win += 1
            else:
                self.lose += 1
        else:
            self.draw += 1
    
    def merge_with(self, other:'ResultCounter'):
        self.win += other.win
        self.draw += other.draw
        self.lose += other.lose
    
    def inverse(self):
        return ResultCounter(self.lose, self.draw, self.win)
    
    def __str__(self):
        return f"Win: {self.win}, Draw: {self.draw}, Lose: {self.lose}"