# -*- coding: utf-8 -*-

"""
dsGameSolver to compute Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2018-2020  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np






def y0_qre(u, phi, nums_a):
    ## linear system of equations
    
    num_s, num_p = u.shape[0:2]
    strategyAxes = tuple(np.arange(1, 1+num_p))
    
    ## strategies: all players randomize uniformly
    beta = np.concatenate( [np.log(np.ones(nums_a[s,p])/nums_a[s,p]) for s in range(num_s) for p in range(num_p)] )
    
    ## state values: solve linear system of equations for each player
    V = np.nan * np.ones(num_s*num_p, dtype=np.float64)
    for p in range(num_p):
        A = np.identity(num_s) - np.nanmean(phi[:,p], axis=strategyAxes)
        b = np.nanmean(u[:,p], axis=strategyAxes)
        mu_p = np.linalg.solve(A, b)
        for s in range(num_s):
            V[s*num_p+p] = mu_p[s]
    
    ## same computation, but also works for strategy profiles other than centroid
    ## (needs T_y2beta)
    #import string
    #beta = np.concatenate( [np.log(np.ones(nums_a[s,p])/nums_a[s,p]) for s in range(num_s) for p in range(num_p)] )
    #beta_array_withNaN = np.einsum('spaN,N->spa', T_y2beta, beta)
    #sigma_array_withNaN = np.exp(beta_array_withNaN)
    #sigma_array = sigma_array_withNaN.copy()
    #sigma_array[np.isnan(sigma_array)] = 0
    #sigma_p_list = [sigma_array[:,p] for p in range(num_p)]
    #einsum_formula = 'sp' + string.ascii_uppercase[0:num_p] + ',s' + ',s'.join(string.ascii_uppercase[0:num_p]) + '->sp'
    #u_sigma = np.einsum(einsum_formula, u, *sigma_p_list)
    #einsum_formula = 'sp' + string.ascii_uppercase[0:num_p] + 't,s' + ',s'.join(string.ascii_uppercase[0:num_p]) + '->spt'
    #phi_sigma = np.einsum(einsum_formula, phi, *sigma_p_list)
    #V = np.nan * np.ones(num_s*num_p, dtype=np.float64)
    #for p in range(num_p):
    #    A = np.identity(num_s) - phi_sigma[:,p]
    #    b = u_sigma[:,p]
    #    mu_p = np.linalg.solve(A, b)
    #    for s in range(num_s):
    #        V[s*num_p+p] = mu_p[s]
    
    y0 = np.concatenate([beta, V, [0.0]])
    
    if np.isnan(y0).any():
        print('Error: Linear system of equations could not be solved.')
        return False, y0
    
    else:
        return True, y0






## ============================================================================
## End of script
## ============================================================================
