from env.tictactoe.tictactoe_env import TicTacToeGame 


# chechk step
b = TicTacToeGame()
b.reset()
b.step(0)
b.step(1)
b.step(2)
_, _, done = b.step(3)
assert done == False
obs = b.observation

assert obs[0][0] == 1
assert obs[0][1] == -1
assert obs[0][2] == 1
assert obs[1][0] == -1

# check win
b.reset()
b.step(0)
b.step(3)
b.step(1)
b.step(4)
_, r, done = b.step(2)
assert done == True
assert r == 1


# check draw
b.reset()
b.step(0)
b.step(1)
b.step(2)
b.step(3)
b.step(4)
b.step(8)
b.step(5)
b.step(6)
_, r, done = b.step(7)
assert done == True
assert r != 0 and abs(r) < 1


print("PASSED!")