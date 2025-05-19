import numpy as np

from env.base_env import BaseGame

INF = 1e16

class MCTSNode:
    def __init__(
        self, 
        action:int, env:BaseGame, reward:float,
        parent: 'MCTSNode' = None
    ) -> None:
        self.env  = env
        
        self.action = action
        self.reward = reward
        self.n_action = env.action_space_size
        self.action_mask = env.action_mask
        
        self.parent = parent
        self.children = {} # a dictionary of ( action -> MCTSNode )
        
        self.child_V_total = np.zeros(self.n_action, dtype=np.float32)
        self.child_N_visit = np.zeros(self.n_action, dtype=np.int32)
        self.child_priors  = np.zeros(self.n_action, dtype=np.float32)
        
    @property
    def done(self):
        return self.env.ended
    
    @property
    def child(self, action):
        return self.children.get(action, None)
    
    def add_child(self, action:int):
        new_env = self.env.fork()
        _, new_reward, _ = new_env.step(action)
        child = MCTSNode(action=action, env=new_env, reward=new_reward, parent=self)
        self.children[action] = child
        return child
        
    def get_child(self, action:int) -> 'MCTSNode':
        return self.children.get(action, None)
    
    def set_prior(self, prior:np.ndarray):
        self.child_priors = prior.copy()
    
    def has_child(self, action:int):
        return action in self.children
        
    def cut_parent(self):
        self.parent = None
    
    
        