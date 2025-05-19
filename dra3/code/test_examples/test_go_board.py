from env.go.goboard import GoBoard as Board
import numpy as np

'''
This is a test script for GoBoard
Note: Read the print instructions at the end of this file!
'''
board = Board(5) 

# capture test
board.load_numpy(np.array([
    [ 0, 0,-1, 0, 0],
    [ 0,-1, 1,-1, 0],
    [-1, 1, 0, 1,-1],
    [ 0,-1, 1,-1, 0],
    [ 0, 0,-1, 0, 0],
]))
# print(board.__str__())
assert set(board.get_legal_moves( 1))==set([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 4), (3, 0), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)])
board.add_stone(2, 2,-1)
# print(board.__str__())
assert (board.to_numpy()==np.array([
    [ 0, 0,-1, 0, 0],
    [ 0,-1, 0,-1, 0],
    [-1, 0,-1, 0,-1],
    [ 0,-1, 0,-1, 0],
    [ 0, 0,-1, 0, 0],
])).all()
assert set(board.get_legal_moves( 1))==set([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 4), (3, 0), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)])
assert set(board.get_legal_moves(-1))==set([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 2), (1, 4), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)])
board.load_numpy(np.array([
    [ 0,-1, 1,-1, 0],
    [-1, 1, 1, 1,-1],
    [ 1, 1, 1, 1, 1],
    [ 1, 1, 1, 1, 0],
    [ 1, 1, 1,-1, 0],
]))
# print(board.__str__())
assert set(board.get_legal_moves( 1))==set([(0, 0), (0, 4), (3, 4), (4, 4)])
assert set(board.get_legal_moves(-1))==set([(3, 4), (4, 4)])
board.add_stone(3, 4,-1)
# print(board.__str__())
assert (board.to_numpy()==np.array([
    [ 0,-1, 0,-1, 0],
    [-1, 0, 0, 0,-1],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0,-1],
    [ 0, 0, 0,-1, 0],
])).all()


# ko-rule test
board.load_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
]))
assert set(board.get_legal_moves(-1))==set([(0, 0), (0, 3), (0, 4), (1, 1), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
assert set(board.get_legal_moves(1)) ==set([(0, 3), (0, 4), (1, 1), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
# print(board.__str__())
board.add_stone(1, 1, 1)
# print(board.__str__())
assert (board.to_numpy()==np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 1, 0, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
])).all()
assert set(board.get_legal_moves(-1))==set([(0, 0), (0, 3), (0, 4), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
assert set(board.get_legal_moves(1)) ==set(board.get_legal_moves(-1))
board.add_stone(1, 4, -1)
board.add_stone(2, 3, 1)
# print(board.__str__())
assert set(board.get_legal_moves(-1))==set([(0, 0), (0, 3), (0, 4), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
assert set(board.get_legal_moves(1)) ==set(board.get_legal_moves(-1))
board.add_stone(1, 2, -1)
assert (board.to_numpy()==np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1,-1],
    [ 0,-1, 1, 1, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
])).all()

# score test
board = Board(5) 
board.load_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
]))
s = board.get_score()
assert s == [4, 6]
board.add_stone(1, 1, 1)
# print(board.__str__())
assert board.get_score() == [6, 4]

board.load_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0, 0, 1, 0],
    [ 0,-1, 0, 1, 0],
    [ 0,-1, 0, 1, 0],
    [ 0, 0,-1, 0, 1],
]))
# print(board.__str__())
assert board.get_score() == [10, 10]

board.load_numpy(np.array([
    [ 0,-1, 1,-1, 0],
    [-1, 0, 1, 1,-1],
    [ 0, 1, 0, 1, 0],
    [ 1, 0, 1, 1, 0],
    [ 0, 1,-1, 0, 1],
]))
# print(board.__str__())
assert board.get_score() == [13, 7]


board = Board(9)
board.load_numpy(np.array([
 [ 1, 1, 1, 1, 1,-1,-1,-1, 0],
 [ 1, 1, 1, 0, 1,-1,-1,-1,-1],
 [ 1, 1, 1, 1, 1,-1,-1,-1,-1],
 [ 1, 1, 1, 1, 1, 1, 1,-1,-1],
 [-1, 1, 1, 1, 1,-1, 1,-1,-1],
 [-1,-1, 1,-1,-1,-1, 1,-1, 0],
 [-1,-1,-1,-1, 1,-1,-1,-1,-1],
 [ 1,-1, 1, 1, 1,-1,-1,-1,-1],
 [ 1, 1, 1, 0, 1,-1, 0,-1,-1],
]))
assert set(board.get_legal_moves(-1))==set([(0, 8), (1, 3), (5, 8), (8, 3), (8, 6)])
assert len(board.get_legal_moves(1))==0
assert board.get_score() == [39, 42]

board = Board(9)
board.load_numpy(np.array([
 [ 1, 1, 1, 1, 1,-1,-1,-1, 0],
 [ 1, 0, 0, 0, 1,-1,-1,-1,-1],
 [ 0, 0, 0, 0, 1,-1,-1,-1,-1],
 [ 1, 0, 0, 1, 0, 1, 1,-1,-1],
 [-1, 1, 1, 1, 1,-1, 1,-1,-1],
 [-1,-1, 1,-1,-1,-1, 1,-1, 0],
 [-1,-1,-1,-1, 1,-1,-1,-1,-1],
 [ 1,-1, 1, 1, 1,-1,-1,-1,-1],
 [ 1, 1, 1, 0, 1,-1, 0,-1,-1],
]))
assert board.get_score() == [39, 42]

board.load_numpy(np.array([
[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
[ 1, 1, 1, 1, 1,-1, 0, 1, 0],
[ 1, 1, 1, 1, 1, 1, 1, 1, 1],
[ 1,-1, 0, 1, 1,-1, 1, 1, 0],
[ 1, 1, 1, 1, 1,-1,-1, 1, 1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1, 0,-1,-1,-1,-1,-1,-1],
[-1,-1, 1,-1,-1,-1,-1, 0,-1],
[ 0,-1,-1,-1,-1,-1,-1, 0,-1],
]))
print(board.get_score())


print("TEST PASSED! yeah~")
print("Note: This test is not complete, you should (at least) test the following cases yourself:")
print("      1. Ending condition")
print("      2. Wining condition")
