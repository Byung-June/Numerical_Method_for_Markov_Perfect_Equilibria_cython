# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from dsGameSolver.gameSolver import dsSolve

alpha = 0.25   ## productivity parameter
beta = 0.95    ## discount factor
gamma = 2      ## relative risk aversion
delta = 0      ## capital depreciation

## utility function
def u(c):
    return (c**(1-gamma)) / (1-gamma)

## production function
def f(k): 
    return ((1-beta)/(alpha*beta)) * k**alpha

## state space K[a] (capital stock)
k_min, k_max, k_step = 0.4, 1.6, 0.1
num_k = int(1 + (k_max-k_min) / k_step)
K = np.linspace(k_min, k_max, num_k)

## action space C[s,a] (consumption)
    ## state dependent to let c_{t}(k_{t}) -> k_{t+1}
    ## with restriction c_{t} >= 0
C = np.nan * np.ones((num_k,num_k))
for s in range(num_k):
    C[s] = (1-delta)*K[s] + f(K[s]) - K
C[C < 0] = np.nan

## numbers of actions in each state
nums_a = np.zeros(num_k, dtype=np.int32)
for s in range(num_k):
    nums_a[s] = len(C[s][~np.isnan(C[s])])

payoffMatrices = []
for s in range(num_k):
    payoffMatrix = np.nan * np.ones((1, nums_a[s]))
    for a in range(nums_a[s]):
        payoffMatrix[0,a] = u(C[s,a])
    payoffMatrices.append( payoffMatrix )

transitionMatrices = []
for s in range(num_k):
    transitionMatrix = np.zeros((nums_a[s], num_k))
    for a in range(nums_a[s]):
        for s_ in range(num_k):
            if a == s_:
                transitionMatrix[a,s_] = 1
    transitionMatrices.append( transitionMatrix )

equilibrium = dsSolve(
        payoffMatrices, transitionMatrices, beta, 
        showProgress=True, plotPath=True)

# Dynamic stochastic game with 13 states, 1 players and 109 actions.
# Initial value for homotopy continuation successfully found.
# ==================================================
# Start homotopy continuation
# Step 7006:   t = 29629.89,   s = 121765.69,   ds = 1000.00   
# Final Result:   max|y-y_|/ds = 0.0E+00,   max|H| = 4.9E-09
# Time elapsed = 0:01:37
# End homotopy continuation
# ==================================================

policies = np.nan * np.ones(num_k)
values = np.nan * np.ones(num_k)
for s in range(num_k):
    ## get optimal actions from pure-strategy equilibrium
    a = np.where((np.round(equilibrium['strategies'][s,0]) == 1))[0]
    policies[s] = C[s, a]
    values[s] = equilibrium['stateValues'][s,0]

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1.set_title('Policy Function')
ax1.set_xlabel(r'capital stock $k_{t}$')
ax1.set_ylabel(r'consumption $c_{t}$')
ax1.plot(K, policies)
ax1.grid()
ax2 = fig.add_subplot(122)
ax2.set_title('Value Function')
ax2.set_xlabel(r'capital stock $k_{t}$')
ax2.set_ylabel(r'present value of utility $V(k_{t})$')
ax2.plot(K, values)
ax2.grid()
plt.show()




"""
## get plot of policy and value functions
fig.savefig('OptimalGrowthModel_functions.pdf', bbox_inches='tight')

## get plot of path
from dsGameSolver.gameClass import dsGame
game = dsGame(payoffMatrices, transitionMatrices, beta)
game.init()
game.solve(trackingMethod='normal', showProgress=True)
fig = game.plot()
fig.savefig('OptimalGrowthModel_path.pdf', bbox_inches='tight')
"""






## ============================================================================
## end of file
## ============================================================================