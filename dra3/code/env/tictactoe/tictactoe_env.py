from ..base_env import *
from .tictactoeboard import TicTacToeBoard
# from .tictactoe_board_python import TicTacToeBoard

from copy import deepcopy

CROSS, CIRCLE = BLACK, WHITE

class TicTacToeGame(BaseGame):
    N = 3
    def __init__(self):
        BaseGame.__init__(self, TicTacToeGame.N, TicTacToeGame.N)
        self.board:TicTacToeBoard = None
        self._action_size = TicTacToeGame.N**2
        self._current_player = CROSS
        self._ended = False
        self._action_mask_cache = None
    
    def init_param_list(self):
        return []
    
    def _coord2actionid(self, x, y):
        return x*self.m + y

    def _actionid2coord(self, action_id):
        return action_id // self.m, action_id % self.m
    
    @property
    def action_mask(self):
        return self._action_mask_cache
    
    def update_action_mask(self):
        valid_actions = self.board.get_legal_moves()
        mask = np.zeros(self._action_size)
        for x, y in valid_actions:
            mask[self._coord2actionid(x, y)] = 1
        self._action_mask_cache = mask
    
    @property
    def observation(self):
        return self.board.to_numpy()
        # return self.board.pieces
    
    def get_canonical_form_obs(self):
        return self.compute_canonical_form_obs(self.observation, self._current_player)

    def compute_canonical_form_obs(self, obs:np.ndarray, player:int)-> np.ndarray:
        return obs * player
    
    def fork(self):
        game = type(self)()
        if self.board is not None:
            game.board = self.board.copy()
        if self._action_mask_cache is not None:
            game._action_mask_cache = self._action_mask_cache.copy()
        game._current_player = self.current_player
        # game.board = deepcopy(self.board)
        game._copy_basic_info(self)
        return game
        
    def reset(self, return_obs=True):
        # faster if return_obs=False 
        self._ended = False
        self._current_player = CROSS
        self.board = TicTacToeBoard(self.n, self.n, self.n) if self.board is None else self.board.reset()
        self.update_action_mask()
        return self.observation if return_obs else None
    
    def step(self, action:int, return_obs=True):
        # faster if return_obs=False 
        self._check_reset()
        if self._action_mask_cache[action] == 0:
            raise ValueError(f"Invalid action: {action}")
        self.board.execute_move(self._actionid2coord(action), self._current_player)
        self.update_action_mask()
        reward = self._get_winner() * self.current_player
        self._ended = not (reward == NOTEND)
        self._current_player = -self._current_player
        return self.observation if return_obs else None, reward, self._ended
    
    def _get_winner(self):
        winner = self.board.check_winner()
        if winner != NOTEND:
            return winner
        if not self.board.has_legal_moves():
            return DRAW
        return NOTEND
        # if self.board.check_win_status(CIRCLE):
        #     return CIRCLE
        # if self.board.check_win_status(CROSS):
        #     return CROSS
        # if not self.board.has_legal_moves():
        #     return DRAW
        # return NOTEND
    
    def to_string(self):
        b = self.observation
        ret = ""
        to_token = lambda x: "X" if x == CROSS else "O" if x == CIRCLE else "-"
        for i in range(self.n):
            ret += f"{' '.join(to_token(b[i, j]) for j in range(self.m))}\n"
        return ret