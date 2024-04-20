#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:31:26 2023

@author: david
"""
#%%

import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime
import sympy as sp
from sympy.physics.quantum.dagger import Dagger as dgr
from joblib import delayed, Parallel
import pylab as plb

plb.rcParams['font.size'] = 45
plt.rcParams["figure.figsize"] = (18,12)
'''
#1 create 1-qubit initial state vector
#2 create initial noiseless single qubit gate
#3 calculate deterministic part of noise
#4 calculate stochastic part of noise
#5 modify gate
#6 evolution of state
#7 get single solution
#8 get averaged solution
#9 visualization
'''

'''
-e^ikr is for the position of the laser focus w.r.t. position of the atom...
for friday (oct 6th) do for both rydberg and superconducting single-qubit evolution with and without stochastic part and compare....
'''
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        

class single_noisy_gate():
    def __init__(self, theta, phi, K_array, gamma):
        '''
        Initialize run by giving the angles for single qubit gate
        '''
        self.theta = theta
        self.phi = phi
        self.K_array = K_array
        self.gamma = gamma
    
    def single_qubit_gate_ryd(self):
        
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        U = sp.Matrix(([sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) - 1J*d/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(-1J*o*x1/o_p*sp.sin(o_p/2*t)), 0, 0]
                          ,[sp.exp(1J*d*t/2)*(-1J*o*x2/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) + 1J*d/o_p*sp.sin(o_p/2*t)), 0, 0]
                          ,[0,0,1,0]
                          ,[0,0,0,1]))
        
        return U
    
    def __det_part_r(self, K_array, params):
               
        U = self.single_qubit_gate_ryd()
        
        lam = []
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        for i in range(len(K_array)):
            gamma = self.gamma[i]
            
            L = (dgr(U)*K_array[i][0]*U)
            tmp1 = -gamma*1/2*(dgr(L)*L - L*L)
            tmp2 = sp.integrate(tmp1, (t, 0, 1))
            tmp3 = sp.lambdify([o_p, d, o, x1, x2], tmp2, "numpy")
            
            lam.append(tmp3(*params))
            
        lam_tot = np.sum(lam, axis = 0)
        
        return lam_tot
    
    def __stochmat_r(self, params):
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        V = [sp.sin(o_p*t/2), sp.cos(o_p*t/2), sp.sin(d*t/2), sp.cos(d*t/2)]
        
        R = [V[1]*V[3], V[0]*V[2], V[0]*V[3], V[2]*V[3]]

        
        var_11 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[0]*R[0], (t, 0, 1)), 'numpy')
        var_12 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[0]*R[1], (t, 0, 1)), 'numpy')
        var_22 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[1]*R[1], (t, 0, 1)), 'numpy')
        
        var_33 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[2]*R[2], (t, 0, 1)), 'numpy')
        var_34 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[2]*R[3], (t, 0, 1)), 'numpy')
        var_44 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[3]*R[3], (t, 0, 1)), 'numpy')
        
        var_11 = var_11(*params)
        var_12 = var_12(*params)
        var_22 = var_22(*params)

        var_33 = var_33(*params)
        var_34 = var_34(*params)
        var_44 = var_44(*params)
        
        st_mat = np.array([[var_11, var_12, 0, 0],[var_12, var_22, 0, 0], [0,0, var_33, var_34], [0,0, var_34, var_44]])
        
        return st_mat
    
    def __stochmat_r_deph(self, params):
        '''
        stochastic matrix for dephasing term, lindblad = Z
        '''
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        V = [sp.sin(o_p*t/2), sp.cos(o_p*t/2)]
        
        R = [V[0]*V[0], V[1]*V[1], sp.cos(o_p*t), 1, sp.sin(o_p*t)]
        
        var_11 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[0]*R[0], (t, 0, 1)), 'numpy')
        var_12 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[0]*R[1], (t, 0, 1)), 'numpy')
        var_13 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[0]*R[2], (t, 0, 1)), 'numpy')
        var_14 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[0]*R[3], (t, 0, 1)), 'numpy')

        var_22 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[1]*R[1], (t, 0, 1)), 'numpy')
        var_23 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[1]*R[2], (t, 0, 1)), 'numpy')
        var_24 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[1]*R[3], (t, 0, 1)), 'numpy')
        
        var_33 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[2]*R[2], (t, 0, 1)), 'numpy')
        var_34 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[2]*R[3], (t, 0, 1)), 'numpy')
        var_44 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[3]*R[3], (t, 0, 1)), 'numpy')
        
        var_55 = sp.lambdify([o_p, d, o, x1, x2], sp.integrate(R[4]*R[4], (t, 0, 1)), 'numpy')

        var_11 = var_11(*params)
        var_12 = var_12(*params)
        var_13 = var_13(*params)
        var_14 = var_14(*params)
        
        var_22 = var_22(*params)
        var_23 = var_23(*params)
        var_24 = var_24(*params)
        
        var_33 = var_33(*params)
        var_34 = var_34(*params)
        var_44 = var_44(*params)

        var_55 = var_55(*params)
    
        st_mat = np.array([[var_11, var_12, var_13, var_14, 0],[var_12, var_22, var_23, var_24, 0], [var_13, var_23, var_33, var_34, 0], [var_14, var_24, var_34, var_44, 0], [0, 0, 0, 0, var_55]])
        
        return st_mat

    def __stoch_part_r(self, K_array, params, st_mat, st_mat_deph):
        '''
        order of params: o_p, d, o, x1, x2
        '''
        
        g_1 = self.gamma[0]
        gd = self.gamma[1]

        
        o_p, d, o, x1, x2 = params

        sample = np.random.multivariate_normal(np.array([0,0,0,0]), st_mat, 1)
        st_arr = np.array([sample[0][i] for i in range(0,4)])
        
        val_1 = np.sqrt(g_1)*(-(1J*params[1]*st_arr[2] - params[1]*st_arr[1])/params[0] + st_arr[0] + 1J*st_arr[3])
        val_2 = -(np.sqrt(g_1)*params[2]*params[3]/params[0])*(1J*st_arr[2] - st_arr[1])
        
        sample = np.random.multivariate_normal(np.array([0,0,0,0]), st_mat, 1)
        st_arr = np.array([sample[0][i] for i in range(0,4)])
        
        val_3 = -(np.sqrt(g_1)*params[2]*params[4]/params[0])*(1J*st_arr[2] - st_arr[1])
        val_4 = np.sqrt(g_1)*((1J*params[1]*st_arr[2] - params[1]*st_arr[1])/params[0] + st_arr[0] + 1J*st_arr[3])
        
        M = np.zeros([4,4])
        M = M + 1J*M
        
        M[3,0] = val_1
        M[3,1] = val_2
        
        M2 = np.zeros([4,4])
        M2 = M2 + 1J*M2
        
        M2[3,0] = val_3
        M2[3,1] = val_4
        
        sample = np.random.multivariate_normal(np.array([0,0,0,0,0]), st_mat_deph, 1)
        st_arr = np.array([sample[0][i] for i in range(0,5)])

        val_1 = np.sqrt(gd)*(d**2*st_arr[0] - o**2*st_arr[0] + o_p**2*st_arr[1])/(o_p**2)
        val_4 = np.sqrt(gd)*(-d**2*st_arr[0] + o**2*st_arr[0] - o_p**2*st_arr[1])/(o_p**2)
        
        val_2 = 0.5*np.sqrt(gd)*o*(-x1-x2)*((d*st_arr[2] - d*st_arr[3] + o_p*1J*st_arr[4]))/(o_p**2)
        val_3 = 0.5*np.sqrt(gd)*o*(-x2-x1)*((d*st_arr[2] - d*st_arr[3] - o_p*1J*st_arr[4]))/(o_p**2)
                
        Md = np.zeros([4,4])
        Md = Md + 1J*Md
        
        Md[0, 0] = val_1
        Md[0, 1] = val_2
        Md[1, 0] = val_3
        Md[1, 1] = val_4

        return 1J*(M + M2 + Md)
    
    def __modify_gate_s(self, params = [1,1,1,1,1,1], K_array = None, st_mat = None, st_mat_deph = None, det_part = None, U_r = None):
        
        U = U_r
        params = params[1:]
        result = U @ scipy.linalg.expm(det_part)@ scipy.linalg.expm(self.__stoch_part_r(K_array, params, st_mat, st_mat_deph))
        
        return result
    
    def singlegate_runs(self, psi_0, N, shots, params):
        '''                
        The actual run function.
        psi_0 > has to be for single qubit
        N >  number of single qubit gates.
        '''
        
        K_array = self.K_array
        
        results_st1 = np.zeros([N, shots])
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        tmp = sp.lambdify([t, o_p, d, o, x1, x2], self.single_qubit_gate_ryd(), "numpy")
        U = tmp(*params)
        
        det_part = self.__det_part_r(K_array, params[1:])
        
        st_mat = self.__stochmat_r(params[1:])
        
        for j in range(0,shots):
            U_array = np.array([self.__modify_gate_s(params = params, st_mat = st_mat, det_part = det_part, U_r = U) for i in range(N)])
            for i in range(N):
                if i == 0:
                    res = np.matmul(U_array[-i-1], psi_0)
                    tmp = (np.outer(np.conj(res), res))
                    results_st1[i][j] = np.real(tmp[0][0])
                else:
                    res = np.matmul(U_array[-i-1], res)
                    tmp = (np.outer(np.conj(res), res))
                    results_st1[i][j] = np.real(tmp[0][0])            
        
        results_st2 = np.array([np.sum(results_st1[k])/shots for k in range(N)])
        
        return(results_st2)
    
    def singlequbit_single_run(self, psi_0, N, params):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        
        K_array = self.K_array
        
        results_p0 = np.zeros([N])
        results_p1 = np.zeros([N])
        results_pd = np.zeros([N])
        
        results_p0[0] = psi_0[0]
        results_p1[0] = psi_0[1]
        results_pd[0] = psi_0[3]

        
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        tmp = sp.lambdify([t, o_p, d, o, x1, x2], self.single_qubit_gate_ryd(), "numpy")
        U = tmp(*params)
        
        st_mat = self.__stochmat_r(params[1:])
        st_mat_deph = self.__stochmat_r_deph(params[1:])
        det_part = self.__det_part_r(K_array, params[1:])
        
        U_array = np.array([self.__modify_gate_s(params = params, st_mat = st_mat, det_part = det_part, U_r = U, st_mat_deph = st_mat_deph) for i in range(N)])
        for i in range(1,N):
            if i == 1:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = np.outer(np.conj(res), res)
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_pd[i] = np.real(tmp[3][3])
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = np.outer(np.conj(res), res)
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_pd[i] = np.real(tmp[3][3])
        
        return(results_p0, results_pd)
    
    def singlequbit_sample_runs(self, psi_0, N, shots, params):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        with tqdm_joblib(tqdm(desc="My calculation", total=shots)) as progress_bar:
            res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N, params) for i in range(shots))
                
        res = np.array(res).sum(axis=0)/(shots)
        
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        
        return(res)