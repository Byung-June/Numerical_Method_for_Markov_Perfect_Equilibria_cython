# -*- coding: utf-8 -*-

"""
dsGameSolver: Computing Markov perfect equilibria of dynamic stochastic games.
Copyright (C) 2018-2020  Steffen Eibelshäuser & David Poensgen

This program is free software: you can redistribute it 
and/or modify it under the terms of the MIT License.
"""






import numpy as np
#np.seterr(over='ignore')
#np.set_printoptions(precision=6, suppress=True, threshold=1000)
#np.set_printoptions(threshold=np.nan)

import matplotlib.pyplot as plt
import time
import datetime
import sys


import dsGameSolver.homCont_subfunctions as func






class hist:
    
    def __init__(self, dim_y, sigma_count=-1, y2sigma_fun=None, cutoff=500000):
        self.cutoff = cutoff
        self.y2sigma_fun = y2sigma_fun
        self.sigma_count = sigma_count
        self.t = np.ones(self.cutoff) * np.nan
        self.s = np.ones(self.cutoff) * np.nan
        self.y = np.ones((self.cutoff, dim_y)) * np.nan
        self.sign = np.ones(self.cutoff) * np.nan
        self.cond = np.ones(self.cutoff) * np.nan
        self.finalized = False
    
    def update(self, t, s, y, sign, cond):
        nan_indices = np.where(np.isnan(self.t))[0]
        if len(nan_indices) > 0:
            j = nan_indices[0]
            self.t[j] = t
            self.s[j] = s
            self.y[j,:] = y
            self.sign[j] = sign
            self.cond[j] = cond
    
    def finalize(self):
        self.t = self.t[~np.isnan(self.t)]
        self.s = self.s[~np.isnan(self.s)]
        self.y = self.y[~np.isnan(self.y).any(axis=1), :]
        self.sign = self.sign[~np.isnan(self.sign)]
        self.cond = self.cond[~np.isnan(self.cond)]
        if self.y2sigma_fun is not None and not self.finalized:
            self.y[:, :self.sigma_count] = self.y2sigma_fun(self.y[:, :self.sigma_count])
        self.finalized = True
    
    def plotprep(self, cutoff=100000, samplefreq=10):
        if not self.finalized:
            self.finalize()
        while len(self.t) > cutoff:
            ## downsample
            self.t = self.t[::samplefreq]
            self.s = self.s[::samplefreq]
            self.y = self.y[::samplefreq, :]
            self.sign = self.sign[::samplefreq]
            self.cond = self.cond[::samplefreq]
    
    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ## s -> t
        ax1 = fig.add_subplot(221)
        ax1.set_title('Homotopy Path')
        ax1.set_xlabel(r'path length $s$')
        ax1.set_ylabel(r'homotopy parameter $t$')
        ax1.set_ylim(0, np.max([1, np.amax(self.t)]))
        ax1.plot(self.s, self.t)
        ax1.grid()
        ## s -> y
        ax2 = fig.add_subplot(222)
        ax2.set_title('Strategy Convergence I')
        ax2.set_xlabel(r'path length $s$')
        ax2.set_ylabel(r'strategies $\sigma$')
        ax2.set_ylim(0,1)
        ax2.plot(self.s, self.y[:, :self.sigma_count])
        ax2.grid()
        ## s -> cond(J)
        ax3 = fig.add_subplot(223)
        ax3.set_title('Numerical Stability')
        ax3.set_xlabel(r'path length $s$')
        ax3.set_ylabel(r'condition number $cond(J)$')
        ax3.plot(self.s, self.cond)
        ax3.grid()
        ## t -> y
#        ax4 = fig.add_subplot(224)
#        ax4.set_title('Strategy Convergence II')
#        ax4.set_xlabel(r'homotopy parameter $t$')
#        ax4.set_ylabel(r'strategies $\sigma$')
#        ax4.set_ylim(0,1)
#        ax4.plot(self.t, self.y[:, :sigma_count])
#        ax4.grid()
        ax4 = fig.add_subplot(224)
        ax4.set_title('Path Direction')
        ax4.set_xlabel(r'path length $s$')
        ax4.set_ylabel(r'sign of tangent')
        ax4.set_ylim(-1.5,1.5)
        ax4.plot(self.s, self.sign)
        ax4.grid()
        plt.tight_layout()
        plt.show()
        return fig






def solve(H, J, y0, num_steps=None, s=None, ds=None, sign=None, sigma_count=-1, y2sigma_fun=None,
          hist=None, t_target=np.inf, num_steps_max=np.inf, progress=False,
          **kwargs):
    
    ## start stopwatch
    tic = time.time()
    if progress:
        print('=' * 50)
        print('Start homotopy continuation')
    
    ## default tracking parameters
    parameters = {
            'y_tol': 1e-7,
            't_tol': 1e-7,
            'H_tol': 1e-7,
            'ds0': 0.01,
            'ds_infl': 1.2,
            'ds_defl': 0.5,
            'ds_min': 1e-9,
            'ds_max': 1000,
            'corr_steps_max': 20,
            'corr_dist_max': 0.3,
            'corr_contr_max': 0.3,
            'detJratio_max': 1.3,
            'bifurc_angle_min': 175
            }
    
    ## use **kwargs to overwrite default parameters
    for key in parameters.keys():
        parameters[key] = kwargs.get(key, parameters[key])
    
    ## unpack tracking parameters
    y_tol = parameters['y_tol']
    t_tol = parameters['t_tol']
    H_tol = parameters['H_tol']
    ds0 = parameters['ds0']
    ds_infl = parameters['ds_infl']
    ds_defl = parameters['ds_defl']
    ds_min = parameters['ds_min']
    ds_max = parameters['ds_max']
    corr_steps_max = parameters['corr_steps_max']
    corr_dist_max = parameters['corr_dist_max']
    corr_contr_max = parameters['corr_contr_max']
    detJratio_max = parameters['detJratio_max']
    bifurc_angle_min = parameters['bifurc_angle_min']
    
    ## get orientation of homotopy path
    y_old = y0.copy()
    t_init = y_old[-1]
    t_min = min([t_init, t_target])
    t_max = max([t_init, t_target])
    J_y = J(y_old)
    Q, R = func.QR(J_y)
    if sign is None:
        sign = 1
        dtds = func.tangent(Q, R, sign)[-1]
        if (dtds < 0 and t_target > t_init) or (dtds > 0 and t_target < t_init): 
            sign = - sign
    tangent_old = func.tangent(Q, R, sign)
    detJ_y = np.linalg.det(np.vstack([J_y, tangent_old]))
    
    ## initialize homotopy continuation
    t = t_init
    if num_steps is None:
        num_steps = 0
    if s is None:
        s = 0
    if ds is None:
        ds = ds0
    if hist is not None:
        hist.update(t=t, s=s, y=y0, sign=sign, cond=np.linalg.cond(J_y))
    
    
    ## path tracking loop
    continue_tracking = True
    while continue_tracking:
        
        num_steps += 1
    
        ## compute tangent at y_old
        Q, R = func.QR(J_y)
        tangent = func.tangent(Q, R, sign)
#        tangent = np.zeros(len(y_old))
#        for i in range(len(y_old)):
#            J_y_without_i = np.delete(J_y, i, axis=1)
#            tangent[i] = np.linalg.det(J_y_without_i) * (-1)**(i+1)
#        tangent = -sign * tangent / np.linalg.norm(tangent)
        
        ## test for bifurcation point
        angle = np.arccos( min([1, max([-1, np.dot(tangent, tangent_old)])]) ) * 180 / np.pi
        sign, tangent, angle, sign_swapped, detJ_y = func.sign_testBifurcation(
                sign, tangent, angle, J_y, detJ_y, bifurc_angle_min=bifurc_angle_min, progress=progress)
        
        
        ## predictor-corrector step loop
        success_corr = False
        while not success_corr:
            
            ## predictor
            y_pred = y_old + ds * tangent
            
            ## predictor invalid
            H_y_pred = H(y_pred)
            if np.isnan(H_y_pred).sum() > 0:
                ds = func.deflate(ds=ds, ds_defl=ds_defl)
                if ds < ds_min: break
            
            ## predictor valid
            else:
                
                ## compute J_pinv at predictor point
                J_y_pred = J(y_pred)
                Q, R = func.QR(J_y_pred)
                J_pinv = func.J_pinv(Q, R)
                
                ## corrector loop
                y_corr, J_y_corr, success_corr, corr_steps, corr_contr_init, corr_dist_tot, err_msg = func.y_corrector(
                            y=y_pred, H=H, J=J, H_y=H_y_pred, J_y_pred=J_y_pred, J_pinv=J_pinv,
                            tangent=tangent, ds=ds, sign=sign, 
                            H_tol=H_tol, corr_steps_max=corr_steps_max, corr_dist_max=corr_dist_max, 
                            corr_contr_max=corr_contr_max, detJratio_max=detJratio_max)
                if not success_corr:
                    ds = func.deflate(ds=ds, ds_defl=ds_defl)
                    if ds < ds_min: break
        
        
        ## update parameters
        tangent_old = tangent.copy()
        J_y = J_y_corr.copy()
        t, s, ds, y_old, continue_tracking, success = func.update_parameters(
                s=s, ds=ds, y_corr=y_corr, y_old=y_old,
                corr_steps=corr_steps, corr_contr_init=corr_contr_init, corr_dist_tot=corr_dist_tot,
                ds_infl=ds_infl, ds_defl=ds_defl, ds_min=ds_min, ds_max=ds_max, 
                t_min=t_min, t_max=t_max, t_init=t_init, t_target=t_target, t_tol=t_tol, 
                y_tol=y_tol, sigma_count=sigma_count, progress=progress, err_msg=err_msg)
        
        
        ## print progress report
        cond = np.linalg.cond(J_y)
        if hist is not None:
            hist.update(t=t, s=s, y=y_corr, sign=sign, cond=cond)
        if progress and success_corr and success:
            sys.stdout.write('\rStep {0}:   t = {1:0.4f},   s = {2:0.2f},   ds = {3:0.2f},   cond(J) = {4:0.0f}          '.format( num_steps, t, s, ds, cond ))
            sys.stdout.flush()
        
        if num_steps > num_steps_max:
            continue_tracking = False
            success = False
            if progress:
                sys.stdout.write('\nMaximum number {0} of steps reached.'.format(num_steps_max))
                sys.stdout.flush()
    
    ## end of path tracking loop
    
    
    
    ## output
    if progress:
        H_test = np.max(np.abs(H(y_corr)))
        if y2sigma_fun is not None:
            sigma_test = np.max(np.abs(y2sigma_fun(y_corr[:sigma_count]) - y2sigma_fun(y_old[:sigma_count]))) / ds
        else:
            sigma_test = np.max(np.abs(y_corr[:sigma_count] - y_old[:sigma_count])) / ds
        ## report new step and stop stopwatch
        print('\nFinal Result:   max|dsigma|/ds = {0:0.1E},   max|H| = {1:0.1E}'.format( sigma_test, H_test ))
        print('Time elapsed = {0}'.format( datetime.timedelta(seconds=round(time.time()-tic)) ))
        print('End homotopy continuation')
        print('=' * 50)
    
    output = {
            'success': success,
            'num_steps': num_steps,
            't': t,
            's': s,
            'ds': ds,
            'sign': sign,
            'y_reduced': y_corr
            }
    
    return output






## ============================================================================
## end of file
## ============================================================================
