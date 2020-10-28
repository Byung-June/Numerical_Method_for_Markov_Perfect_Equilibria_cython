# -*- coding: utf-8 -*-


import numpy as np
from dsGameSolver.gameSolver import dsSolve


num_s = 2
num_p = 2
(num_a_min, num_a_max) = (2, 2)
(delta_min, delta_max) = (0.9, 0.95)


nums_a = np.random.randint(low=num_a_min, high=num_a_max+1, size=(num_s,num_p))

payoffMatrices = [np.random.random((num_p, *nums_a[s,:])) for s in range(num_s)]

transitionMatrices = [np.random.exponential(scale=1, size=(*nums_a[s,:], num_s)) for s in range(num_s)]
for s in range(num_s):
    for index, value in np.ndenumerate(np.sum(transitionMatrices[s], axis=-1)):
        transitionMatrices[s][index] *= 1/value

discountFactors = np.random.uniform(low=delta_min, high=delta_max, size=num_p)


equilibrium = dsSolve(payoffMatrices, transitionMatrices, discountFactors,
                      implementationType='ct', 
                      showProgress=True, plotPath=True)

print(np.round(equilibrium['strategies'], 3))
print(np.round(equilibrium['stateValues'], 3))






## ============================================================================
## end of file
## ============================================================================