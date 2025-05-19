from ..base_env import *
from .goboard import GoBoard as Board
import numpy as np
from typing import List, Tuple

class GoGame(BaseGame):
    def __init__(self, n:int=19, obs_mode:str="board"):
        assert n <= MAX_N, f"n should be less than or equal to {MAX_N}, but got {n}"
        assert obs_mode in ["board", "extra_feature"]
        BaseGame.__init__(self, n, n)
        self._action_size = n*n + 1
        self._current_player = BLACK
        self._PASS_ACTION = n*n
        self._valid_action_mask = None
        self._obs_mode = obs_mode
    
    @property
    def obs_mode(self):
        return self._obs_mode
    
    def init_param_list(self):
        return [self.n, self._obs_mode]
        
    def _coord2actionid(self, x, y):
        return x*self.n + y
    
    def _actionid2coord(self, action_id):
        return int(action_id // self.n), int(action_id % self.n)
    
    def _get_winner(self):
        winner = self.board.get_winner()
        return EPS if winner == DRAW else winner

    @property
    def board_array(self):
        return self.board.to_numpy()
    
    @property
    def observation_size(self):
        if self._obs_mode == 'board':
            return self.n, self.n
        elif self._obs_mode == 'extra_feature':
            return (4 * self.n*self.n + 1 + self.action_space_size,)
        else:
            raise NotImplementedError
    
    @property
    def observation(self):
        if self._obs_mode == 'board':
            return self.board.to_numpy()
        elif self._obs_mode == 'extra_feature':
            return np.concatenate([
                self.board.to_numpy().reshape(-1), 
                np.minimum(self.board.get_liberty_map().reshape(-1), 8)/8, 
                self.board.count_neighbor_map(BLACK).reshape(-1)/4, 
                self.board.count_neighbor_map(WHITE).reshape(-1)/4,
                np.ones(1)*self._current_player,
                self.action_mask
                ], axis=0)
        else:
            raise NotImplementedError
    
    @property
    def action_mask(self):
        return self._valid_action_mask

    def get_canonical_form_obs(self):
        # return the observation in the canonical form, 
        # which is converted to the current player's perspective
        # If the current player is BLACK, the observation is returned as it is
        # If the current player is WHITE, the observation is multiplied by -1
        return self.compute_canonical_form_obs(self.observation, self._current_player)
    
    def compute_canonical_form_obs(self, obs:np.ndarray, player:int)-> np.ndarray:
        if self._obs_mode == 'board':
            return obs * player
        elif self._obs_mode == 'extra_feature':
            board_size = self.n*self.n
            return np.concatenate([
                obs[:board_size]*player,
                obs[board_size:board_size*2],
                obs[board_size*2:board_size*3] if player==BLACK else obs[board_size*3:board_size*4],
                obs[board_size*3:board_size*4] if player==BLACK else obs[board_size*2:board_size*3],
                np.ones(1),
                obs[board_size*4+1:]
            ], axis=0)
        
    
    def _update_valid_action_mask(self):
        # update the valid action mask, which is a binary array indicating the valid actions
        # the _valid_action_mask is buffer to avoid computing action_mask every time
        self._valid_action_mask = np.zeros(self._action_size)
        for x, y in self.board.get_legal_moves(self._current_player):
            self._valid_action_mask[self._coord2actionid(x, y)] = 1
        self._valid_action_mask[-1] = 1
    
    def fork(self):
        # copy the current game state
        game = type(self)(self.n)
        if self.board is not None:
            game.board = self.board.copy()
        if self._valid_action_mask is not None:
            game._valid_action_mask = self._valid_action_mask.copy()
        game._current_player = self._current_player
        game._ended = self._ended
        game._obs_mode = self._obs_mode
        return game
        
    def reset(self):
        # reset the game
        BaseGame.reset(self)
        self.board = Board(self.n)
        self._current_player = BLACK
        self._update_valid_action_mask()
        return self.observation
    
    def step(self, action:int, return_obs=True) -> Tuple[np.ndarray, float, bool]:
        assert action < self._action_size, f"Invalid action:{action} for player:{self._current_player}"
        self._check_reset()
        assert self._valid_action_mask[action], f"Invalid action:{action}(coord:{self._actionid2coord(action)}) for player:{self._current_player}"
        # Execute the action, and check if the game is ended
        # @param action: int, action id
        # @return: observation, reward, done
        # NOTE: the reward is 1 when CURRENT player wins, EPS when DRAW, -1 when the opponent player wins, and 0 otherwise
        if action == self._PASS_ACTION:
            self.board.pass_stone(self._current_player)
        else:
            action = self._actionid2coord(action)
            self.board.add_stone(action[0], action[1], self._current_player)
        
        winner = self._get_winner()
        if winner != NOTEND:
            self._ended = True
            reward = self._get_winner() * self.current_player
            self._current_player = -self._current_player
            return self.observation, reward, True
        
        self._current_player = -self._current_player
        self._update_valid_action_mask()
        return self.observation if return_obs else None, 0, False
    
    def to_string(self):
        b = self.board.to_numpy()
        ret = ""
        to_token = lambda x: "○" if x == BLACK else "●" if x == WHITE else "-"
        for i in range(self.n):
            ret += f"{' '.join(to_token(b[i, j]) for j in range(self.m))}\n"
        return ret