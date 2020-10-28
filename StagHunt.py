# -*- coding: utf-8 -*-


import numpy as np
from dsGameSolver.gameSolver import dsSolve

payoffMatrices = [np.array([[[5, 0],
                             [4, 2]],
                            [[5, 4],
                             [0, 2]]])]
equilibrium = dsSolve(payoffMatrices, implementationType='ct')

print(np.round(equilibrium['strategies'],3))
# array([[[0. 1.] 
#         [0. 1.]]])

print(np.round(equilibrium['stateValues'],3))
# array([[2. 2.]])





## ============================================================================
## end of file
## ============================================================================