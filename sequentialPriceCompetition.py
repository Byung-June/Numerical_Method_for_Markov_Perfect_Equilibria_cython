# -*- coding: utf-8 -*-


import numpy as np
np.set_printoptions(precision=2, suppress=True)
import itertools
from dsGameSolver.gameSolver import dsSolve


num_players = 2     ## number of firms
MC = 0              ## marginal costs
d = lambda p: 2-p   ## market demand

## price grid
p_min = 0
p_step = 0.1
p_max = 1 + p_step
num_prices = int(1 + (p_max-p_min) / p_step)
P = np.linspace(p_min, p_max, num_prices)

## profit function
def Pi(p):
    ## number of firms at minimum price, market shares and demand
    N = (p == p.min()).sum()
    shares = [1/N if p_i == p.min() else 0 for p_i in p]
    D = np.array([shares[i] * d(p_i) for i, p_i in enumerate(p)])
    return (p - MC) * D

## state space [(player_to_move, competitor_prices)]
states = []
for i in range(num_players):
    for a_not_i in itertools.product(
            range(num_prices), repeat=num_players-1):
        states.append( (i, np.array(a_not_i)) )
num_states = len(states)
stateIDs = np.arange(num_states)
state_dict = dict(zip(stateIDs, states))

## functions for convenience
def get_state(stateID): 
    return state_dict[stateID]

def get_stateID(state): 
    for s, state_ in state_dict.items():
        if state_ == state:
            return s
    return None

def payoffMatrix(s):
    i, a_not_i = get_state(s)
    ## dimensions of action profile a in state s
        ## player i can choose a price
        ## other players have only one dummy action
    a_dims = np.ones(num_players, dtype=np.int32)
    a_dims[i] = len(P)
    a_dims = tuple(a_dims)
    matrix = np.nan * np.ones( (num_players,) + a_dims )
    for j in range(num_players):
        for a_profile in np.ndindex(a_dims):
            ## insert action of player i into action profile
            a = np.insert(a_not_i, i, a_profile[i])
            prices = P[a]
            matrix[(j,)+a_profile] = Pi(prices)[j]
    return matrix

## vector of transition probabilities given action profile
def transitionProbs(s, a_profile):
    i, a_not_i = get_state(s)
    a = np.insert(a_not_i, i, a_profile[i])
    i_next = (i + 1) % num_players
    a_not_i_next = np.delete(a, i_next)
    s_next = get_stateID((i_next, a_not_i_next))
    probs = np.zeros(num_states)
    probs[s_next] = 1
    return probs

## full transition matrix
def transitionMatrix(s):
    i, a_not_i = get_state(s)
    a_dims = np.ones(num_players, dtype=np.int32)
    a_dims[i] = num_prices
    a_dims = tuple(a_dims)
    matrix = np.nan * np.ones( a_dims + (num_states,) )
    for a in np.ndindex(a_dims):
        matrix[a] = transitionProbs(s, a)
    return matrix

payoffMatrices = [payoffMatrix(s) for s in range(num_states)]
transitionMatrices = [transitionMatrix(s) for s in range(num_states)]

equilibrium = dsSolve(
        payoffMatrices, transitionMatrices, discountFactors=0.95, 
        showProgress=True, plotPath=True)

# Dynamic stochastic game with 24 states, 2 players and 312 actions,
#     in total 288 action profiles.
# Symmetries reduce game to 24 state-player pairs and 156 actions,
#    in total 156 action profiles.
# Initial value for homotopy continuation successfully found.
# ==================================================
# Start homotopy continuation
# Step 83:   t = 1663.83,   s = 9723.90,   ds = 1000.00   
# Final Result:   max|y-y_|/ds = 0.0E+00,   max|H| = 2.4E-09
# Time elapsed = 0:00:21
# End homotopy continuation
# ==================================================


for a2 in range(num_prices):
    print(
            'p2 = {0:0.1f} =>Prob(p1|p2) = {1}'.format( 
                    P[a2], 
                    equilibrium['strategies'][a2,0,:] 
                    )
            )

# p2 = 0.0 =>Prob(p1|p2) = [0.22 0.   0. 0. 0. 0. 0. 0. 0. 0. 0. 0.78]
# p2 = 0.1 =>Prob(p1|p2) = [0.15 0.34 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.51]
# p2 = 0.2 =>Prob(p1|p2) = [0.   1.   0. 0. 0. 0. 0. 0. 0. 0. 0. 0.  ]
# p2 = 0.3 =>Prob(p1|p2) = [0.   1.   0. 0. 0. 0. 0. 0. 0. 0. 0. 0.  ]
# p2 = 0.4 =>Prob(p1|p2) = [0.   1.   0. 0. 0. 0. 0. 0. 0. 0. 0. 0.  ]
# p2 = 0.5 =>Prob(p1|p2) = [0.   1.   0. 0. 0. 0. 0. 0. 0. 0. 0. 0.  ]
# p2 = 0.6 =>Prob(p1|p2) = [0.   0.   0. 0. 0. 1. 0. 0. 0. 0. 0. 0.  ]
# p2 = 0.7 =>Prob(p1|p2) = [0.   0.   0. 0. 0. 0. 1. 0. 0. 0. 0. 0.  ]
# p2 = 0.8 =>Prob(p1|p2) = [0.   0.   0. 0. 0. 0. 0. 1. 0. 0. 0. 0.  ]
# p2 = 0.9 =>Prob(p1|p2) = [0.   0.   0. 0. 0. 0. 0. 0. 1. 0. 0. 0.  ]
# p2 = 1.0 =>Prob(p1|p2) = [0.   0.   0. 0. 0. 0. 0. 0. 0. 1. 0. 0.  ]
# p2 = 1.1 =>Prob(p1|p2) = [0.   0.   0. 0. 0. 0. 0. 0. 0. 0. 1. 0.  ]







"""
## simulate price paths
np.random.seed(123)
T = 30
a_sim = np.zeros((T+1, num_players), dtype=np.int32)   ## paths of actions

## initial prices: firm 1: p_max, firm 2: p_max-p_step, ...
for i in range(num_players):
    a_sim[0,i] = num_prices - 1 - i
    
## simulation
for t in range(T):
    a_sim[t+1,:] = a_sim[t,:]
    i = t % num_players
    a_not_i = np.delete(a_sim[t,:], i)
    s = get_stateID((i, a_not_i))
    a_sim[t+1,i] = np.random.choice(range(num_prices), size=1, 
                                    p=equilibrium['strategies'][s,i,:])[0]
p_sim = P[a_sim]   ## paths of prices


## plot best responses and simulation of price paths
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec

fig = plt.figure(figsize=(12,4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1,1.8])

## 1) best responses
ax1 = fig.add_subplot(gs[0])
ax1.set_title('Best Responses', fontsize=14)
ax1.set_xlabel(r'$p_{2}(p_{1})$', fontsize=12)
ax1.set_ylabel(r'$p_{1}(p_{2})$', fontsize=12)
ax1.set_xlim(P.min()-p_step, P.max()+p_step)
ax1.set_ylim(P.min()-p_step, P.max()+p_step)

## grid
ax1.hlines(P, p_min-1, p_max+1, colors='black', 
           linestyles='dashed', lw=0.5, alpha=0.3)
ax1.vlines(P, p_min-1, p_max+1, colors='black', 
           linestyles='dashed', lw=0.5, alpha=0.3)

## 45° line
ax1.plot([p_min-1, p_max+1], [p_min-1, p_max+1], color='black', 
         linestyle='dotted', lw=1, alpha=1)

## firm 1
for a2 in range(num_prices):
    for a1 in range(num_prices):
        ax1.plot(P[a2], P[a1], alpha=equilibrium['strategies'][a2,0,a1], 
                 linestyle='None', marker='o', markerfacecolor='white', 
                 markeredgecolor='C0', markersize=6)
## firm 2
for a1 in range(num_prices):
    for a2 in range(num_prices):
        ax1.plot(P[a2], P[a1], alpha=equilibrium['strategies'][a1,0,a2], 
                 linestyle='None', marker='x', color='C1', markersize=6)

ax1.legend(handles=[
        Line2D([0], [0], linestyle='None', marker='o', 
               markerfacecolor='white', markeredgecolor='C0', 
               markersize=6, label='firm 1'),
        Line2D([0], [0], linestyle='None', marker='x', color='C1', 
               markersize=6, label='firm 2')
        ], loc=(0.2,0.75))

## 2) price path simulation
ax2 = fig.add_subplot(gs[1])
ax2.set_title('Price Path Simulation', fontsize=14)
ax2.set_xlabel(r'time $t$', fontsize=12)
ax2.set_ylabel(r'price $p_{i,t}$', fontsize=12)
ax2.set_xlim(-1, T+1)
ax2.set_ylim(P.min()-p_step, P.max()+p_step)

ax2.hlines(P, -1, T+1, colors='black', 
           linestyles='dashed', lw=0.5, alpha=0.3)
ax2.vlines(range(0,31,5), p_min-1, p_max+1, colors='black', 
           linestyles='dashed', lw=0.5, alpha=0.3)

ax2.hlines([MC], -1, T+1, colors='black', 
           linestyles='solid', lw=1, alpha=1)
ax2.text(T+1, MC, ' MC', horizontalalignment='left', verticalalignment='center')

ax2.step(range(T+1), p_sim, where='post')
plt.show()
fig.savefig('sequentialPriceCompetition_equilibrium.pdf', bbox_inches='tight')




## get plot of path
from dsGameSolver.gameClass import dsGame
game = dsGame(payoffMatrices, transitionMatrices, 0.95)
game.init()
game.solve(showProgress=True)
fig = game.plot()
fig.savefig('sequentialPriceCompetition_path.pdf', bbox_inches='tight')
"""






## ============================================================================
## end of file
## ============================================================================