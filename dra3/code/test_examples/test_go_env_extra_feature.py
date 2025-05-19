from env import GoGame
import numpy as np

N=7

def print_extra_features(x:np.ndarray, n=None):
    if n is None:
        n=N
    b = n*n
    print(f"Total_size: ", len(x))
    print(f"Player:"+"BLACK" if x[b*4]==1 else "WHITE")
    
    print(f"board_map:")
    print(x[:b].reshape(n, n), '\n')
    print(f"liberty_map:")
    print(x[b:b*2].reshape(n, n), '\n')
    print(f"black_neighbor_map:")
    print(x[b*2:b*3].reshape(n, n), '\n')
    print(f"white_neighbor_map:")
    print(x[b*3:b*4].reshape(n, n), '\n')
    
    print(f"action_map")
    print(x[b*4+1:], '\n')
    
    
# TODO: Add some TEST CASE!
env = GoGame(N, "extra_feature")
obs = env.reset()
print_extra_features(obs)
obs, rwd, is_end = env.step(3*N + 4)
print("[STEP] 4, 5")
print_extra_features(obs)
obs, rwd, is_end = env.step(1*N + 4)
print("[STEP] 2, 5")
obs, rwd, is_end = env.step(2*N + 3)
print("[STEP] 3, 4")
obs, rwd, is_end = env.step(2*N + 5)
print("[STEP] 3, 6")
obs, rwd, is_end = env.step(3*N + 3)
print("[STEP] 4, 4")
print_extra_features(obs)