# -*- coding: utf-8 -*-


import numpy as np
from dsGameSolver.gameSolver import dsSolve

payoffMatrices = [
        ## s = 0
        np.array([
                [[ 0, -1,  1],
                 [ 1,  0, -1],
                 [-1,  1,  0]],
                
                [[ 0,  1, -1],
                 [-1,  0,  1],
                 [ 1, -1,  0]]
                ]), 
        ## s = 1
        np.array([
                [[ 0, -1,  1],
                 [ 1,  0, -1],
                 [-1,  1,  1]],
                
                [[ 0,  1, -1],
                 [-1,  0,  1],
                 [ 1, -1, -1]]
                ]), 
        ## s = 2
        np.array([
                [[ 0, -1,  1],
                 [ 1,  0, -1],
                 [-1,  1, -1]],
                
                [[ 0,  1, -1],
                 [-1,  0,  1],
                 [ 1, -1,  1]]
                ])
    ]

transitionMatrices = [
        ## same for s = 0,1,2
        np.array([
                [[1, 0, 0],
                 [1, 0, 0],
                 [0, 0, 1]],
                
                [[1, 0, 0],
                 [1, 0, 0],
                 [0, 0, 1]],
                
                [[0, 1, 0],
                 [0, 1, 0],
                 [1, 0, 0]]
                ])
    ] * 3

equilibrium = dsSolve(payoffMatrices, transitionMatrices, discountFactors=0.95, 
                      implementationType='ct', plotPath=True)

print(np.round(equilibrium['strategies'],3))
# array([[[0.369 0.298 0.333]
#         [0.369 0.298 0.333]]
#
#        [[0.257 0.409 0.333]
#         [0.480 0.187 0.333]]
#
#        [[0.480 0.187 0.333]
#         [0.257 0.409 0.333]]])

print(np.round(equilibrium['stateValues'],3))
# array([[ 0.     0.   ]
#        [ 0.111 -0.111]
#        [-0.111  0.111]])

print(np.round(equilibrium['homotopyParameter']))
# 6842.0




"""
## get plot of path
from dsGameSolver.gameClass import dsGame
game = dsGame(payoffMatrices, transitionMatrices, discountFactors)
game.solve()
fig = game.plot()
fig.savefig('dynamicRockPaperScissors_path.pdf', bbox_inches='tight')
"""






## ============================================================================
## end of file
## ============================================================================