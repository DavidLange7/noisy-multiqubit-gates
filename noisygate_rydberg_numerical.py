#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:31:26 2023

@author: david
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime
import sympy as sp
from sympy.physics.quantum.dagger import Dagger as dgr
from joblib import delayed, Parallel
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
#%%
class rydberg_twoq_noisy_gate():
    def __init__(self, K_array, omega, delta = 0, V = 0, gamma = []):
        '''
        Initialize run by giving the parameters for single qubit gate
        '''
        self.omega = omega
        self.delta = delta
        self.V = V
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
    
    def single_qubit_gate_ryd_ham(self):
        omega = self.omega
        delta = self.delta
        
        H = np.array([[0, omega/2, 0, 0], 
                      [omega/2, -delta, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        
        return H
            
    def two_qubit_gate_ryd_ham(self):
        '''
        resulting order: 00, 01, 0r, 0d, 10, 11, 1r, 1d, r0, r1, rr, rd, d0, d1, dr, dd
        '''
        omega = self.omega
        delta = self.delta
        V = self.V
        
        sigp = np.kron(np.conjugate([[0,0,1,0]]), ([[0,1,0,0]])).reshape([4,4])
        sigm = np.kron(np.conjugate([[0,1,0,0]]), ([[0,0,1,0]])).reshape([4,4])
        
        n = np.kron(np.conjugate([[0,0,1,0]]), ([[0,0,1,0]])).reshape([4,4])

        H_1 = omega/2*(sigm + sigp) - delta*n
        H_2 = np.kron(H_1, np.identity(4)).reshape([16,16]) + np.kron(np.identity(4), H_1).reshape([16,16]) + V*(np.kron(n, np.identity(4)).reshape([16,16])* np.kron(np.identity(4), n).reshape([16,16]))
        
        return H_2
        
    def __det_part_r(self, ham):

        L = self.K_array
        
        val, vec = np.linalg.eig(-1J*ham)
        v_m1 = (np.linalg.inv(vec))
        t = sp.symbols('t', real = True)

        expr1 = [sp.exp(1J*val[i]*t) for i in range(len(val))]
        expr2 = [sp.exp(-1J*val[i]*t) for i in range(len(val))]

        expr1 = np.diag(expr1)
        expr2 = np.diag(expr2)
        
        lam = []
    
        for ind in range(len(L)):
            L_int = sp.simplify( np.conj(v_m1) @ expr1 @ np.conj(vec) @ L[ind] @ vec @ expr2 @ v_m1 )
            
            tmp1 = sp.Matrix(-1/2*self.gamma[ind]*(np.matmul(np.conj(L_int), L_int) - np.matmul(L_int, L_int)))
            
            tmp2 = sp.integrate(tmp1, (t, 0, 1))
            tmp3 = sp.lambdify(t, tmp2, "numpy")
            lam.append(tmp3(1))
            
        lam_tot = np.sum(lam, axis = 0)

            
        '''
        lam = []
        for i in range(len(self.K_array)):
            L = np.matmul(np.conjugate(U), np.matmul(self.K_array[i], U))

            tmp = -1/2*(np.matmul(np.matrix.conjugate(L), L) - np.matmul(L, L))
            lam.append(tmp)
        lam_tot = np.sum(lam, axis = 0)
        '''
        return lam_tot
    
    def __variance(self, expr):
        '''
        
        Input is a scipy expression involving only "t"-variable
        
        '''
        t = sp.symbols('t', real = True)
        f = sp.lambdify(t, expr**2)
        res = scipy.integrate.quad(f, 0, 1)
        return(res[0])

    def __correlation(self, expr1, expr2):
        '''
        
        Input are scipy expressions involving only "t"-variable
        
        '''
        t = sp.symbols('t', real = True)
        f = sp.lambdify(t, expr1*expr2)
        res = scipy.integrate.quad(f, 0, 1)
        return(res[0])


    def __stats(self, ham):
        '''
        mat is a sympy matrix with only "t"-variable
        '''
        
        corr_rs = []
        corr_ims = []
        
        L = self.K_array
        
        val, vec = np.linalg.eig(-1J*ham)
        v_m1 = np.linalg.inv(vec)
        t = sp.symbols('t', real = True)

        expr1 = [sp.exp(1J*val[i]*t) for i in range(len(val))]
        expr2 = [sp.exp(-1J*val[i]*t) for i in range(len(val))]

        expr1 = np.diag(expr1)
        expr2 = np.diag(expr2)

        for ind in range(len(L)):
    
            tot = sp.simplify( np.conj(v_m1) @ expr1 @ np.conj(vec) @ L[ind] @ vec @ expr2 @ v_m1 )
        
            mat = sp.flatten(tot)
    
            corr_r = np.zeros([len(mat), len(mat)])
            corr_im = np.zeros([len(mat), len(mat)])
            
            for k in range(len(mat)):
                for j in range(len(mat)):
                    binary_1 = mat[k].is_zero
                    binary_2 = mat[j].is_zero
                    
                    if binary_1 != True and binary_2 != True:
                        if k == j:
                            corr_r[k,k] = self.__variance(sp.re(mat[k]))
                        else:
                            corr_r[k,j] = self.__correlation(sp.re(mat[k]), sp.re(mat[j]))
            
            for k in range(len(mat)):
                for j in range(len(mat)):
                    binary_1 = mat[k].is_zero
                    binary_2 = mat[j].is_zero
                    
                    if binary_1 != True and binary_2 != True:
                        if k == j:
                            corr_im[k,k] = self.__variance(sp.re(mat[k]))
                        else:
                            corr_im[k,j] = self.__correlation(sp.re(mat[k]), sp.re(mat[j]))
        
            corr_rs.append(corr_r)
            corr_ims.append(corr_im)
            
        return (corr_rs, corr_ims)
    
    def __stoch_part(self, corr_rs, corr_ims):
        
        len_tot = np.shape(corr_rs[0])[0]
        len_resh = int(np.sqrt(len_tot))
                
        m = np.zeros([len_resh, len_resh]) + 1J*np.zeros([len_resh, len_resh])

        for ind in range(len(self.K_array)):
            corr_r = corr_rs[ind]
            corr_im = corr_ims[ind]
    
            sample_r = np.random.multivariate_normal(np.zeros([len_tot]), corr_r, 1)
            sample_im = np.random.multivariate_normal(np.zeros([len_tot]), corr_im, 1)
    
            s_r = np.array([sample_r[0][i] for i in range(0, len_tot)])
            s_im = np.array([sample_im[0][i] for i in range(0, len_tot)])
            
            #s_r = sample_r[0]
            #s_im = sample_r[0]
          
            
            result = s_r.reshape(len_resh, len_resh) + 1J*s_im.reshape(len_resh, len_resh)
            
            m += np.sqrt(self.gamma[ind])*result

        
        return(1J*m)
            
            
    def __modify_gate_twoq(self, U, det_part, corr_rs = None, corr_ims = None):
                                                   
        result = U @ scipy.linalg.expm(self.__stoch_part(corr_rs, corr_ims)) @ scipy.linalg.expm(det_part)

        return result
    
    def twoqubit_single_run(self, psi_0, N):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        
        results_p0 = np.zeros([N])
        results_p1 = np.zeros([N])
        results_p5 = np.zeros([N])
        results_pd = np.zeros([N])

        results_p0[0] = psi_0[0]
        results_p1[0] = psi_0[1]
        
        results_p5[0] = psi_0[5]
        results_pd[0] = psi_0[-1]

        
        H = self.two_qubit_gate_ryd_ham()
        
        U = scipy.linalg.expm(-1J*H)

        det = self.__det_part_r(H)
        corr_mats = self.__stats(H)
        
        #print(scipy.linalg.expm(self.__stoch_part(corr_mats[0], corr_mats[1])))
        
        U_array = np.array([self.__modify_gate_twoq(U, det_part = det, corr_rs = corr_mats[0], corr_ims = corr_mats[1]) for i in range(N)])
        for i in range(1,N):
            if i == 1:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_p5[i] = np.real(tmp[5][5])
                results_pd[i] = np.real(tmp[-1][-1])

                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_p5[i] = np.real(tmp[5][5])
                results_pd[i] = np.real(tmp[-1][-1])

        
        return(results_p5)
    
    def singlequbit_single_run(self, psi_0, N):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        
        results_p0 = np.zeros([N])
        results_p1 = np.zeros([N])
        results_pd = np.zeros([N])
        
        results_p0[0] = psi_0[0]
        results_p1[0] = psi_0[1]
        
        results_pd[0] = psi_0[3]

        #tmp_st = []
        
        
        H = self.single_qubit_gate_ryd_ham()
        
        U = scipy.linalg.expm(-1J*H)

        det = self.__det_part_r(H)
        corr_mats = self.__stats(H)
        
        #print(scipy.linalg.expm(self.__stoch_part(corr_mats[0], corr_mats[1])))
        
        U_array = np.array([self.__modify_gate_twoq(U, det_part = det, corr_rs = corr_mats[0], corr_ims = corr_mats[1]) for i in range(N)])

        for i in range(1,N):
            if i == 1:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_pd[i] = np.real(tmp[3][3])
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_pd[i] = np.real(tmp[3][3])
                
        return(results_p0)
    
    def twoqubit_sample_runs(self, psi_0, N, shots):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        with tqdm_joblib(tqdm(desc="My calculation", total=shots)) as progress_bar:
            res = Parallel(n_jobs=-1)(delayed(self.twoqubit_single_run)(psi_0, N) for i in range(shots))
        
        #without progress bar: res = Parallel(n_jobs=-1)(delayed(self.twoqubit_single_run)(psi_0, N) for i in range(shots))

        res = np.array(res).sum(axis=0)/(shots)
        
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        
        return(res)
    
    def singlequbit_sample_runs(self, psi_0, N, shots):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        with tqdm_joblib(tqdm(desc="My calculation", total=shots)) as progress_bar:
            res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N) for i in range(shots))
        
        #without progress bar: res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N) for i in range(shots))

        res = np.array(res).sum(axis=0)/(shots)
        
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        
        return(res)