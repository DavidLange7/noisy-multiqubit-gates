#%%
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
#1 create n-qubit initial state vector
#2 create initial noiseless n-qubit gate
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
'''
#%%
class rydberg_noisy_gate():
    def __init__(self, K_array, omega, delta = 0, V = 0, gamma = []):
        '''
        Initialize run by giving the parameters for single qubit gate
        '''
        self.omega = omega
        self.delta = delta
        self.V = V
        self.K_array = K_array
        self.gamma = gamma
        
    def __debugger(self, text, tbp, debug):
        if debug == 2:
            print(text, tbp)
        return None
    
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

        val, vec = np.linalg.eig(ham)
        v_m1 = (np.linalg.inv(vec))

        t = sp.symbols('t', real = True)

        expr1 = [sp.exp((-1J*val[i]*t)) for i in range(len(val))]
        expr2 = [sp.exp(-1J*val[i]*t) for i in range(len(val))]

        expr1 = np.diag(expr1)
        expr1 = expr1.conj()
        expr2 = np.diag(expr2)


       
        lam = []
    
        for ind in range(len(L)):
            L_int =  sp.Matrix(sp.simplify(np.conj(v_m1).T @ expr1 @ np.conj(vec).T @ L[ind] @ vec @ expr2 @ v_m1))
            L_vec = sp.flatten(-1/2*self.gamma[ind]*(np.conj(L_int).T @ L_int - L_int @ L_int))
            
            tmp_tot = np.array([scipy.integrate.quad(sp.lambdify(t, sp.re(L_vec[i]), "numpy"), 0, 1)[0] + 1J*scipy.integrate.quad(sp.lambdify(t, sp.im(L_vec[i]), "numpy"), 0, 1)[0] for i in range(len(L_vec))])

            lam.append(tmp_tot.reshape(int(np.sqrt(len(L_vec))), int(np.sqrt(len(L_vec)))))
            
        lam_tot = np.sum(lam, axis = 0)

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
        
        val, vec = np.linalg.eig(ham)
        v_m1 = np.linalg.inv(vec)
        t = sp.symbols('t', real = True)

        expr1 = [sp.exp((-1J*val[i]*t)) for i in range(len(val))]
        expr2 = [sp.exp(-1J*val[i]*t) for i in range(len(val))]

        expr1 = np.diag(expr1)
        expr1 = expr1.conj()
        expr2 = np.diag(expr2)

        for ind in range(len(L)):
            tot =  sp.simplify(np.conj(v_m1).T @ expr1 @ np.conj(vec).T @ L[ind] @ vec @ expr2 @ v_m1 )
        
            mat = sp.flatten(tot)

            corr_r = np.zeros([len(mat), len(mat)])
            corr_im = np.zeros([len(mat), len(mat)])
            
            for k in range(len(mat)):
                for j in range(len(mat)):
                    binary_1 = t in mat[k].free_symbols
                    binary_2 = t in mat[j].free_symbols
                    
                    if binary_1 == True and binary_2 == True:
                        if k == j:
                            corr_r[k,k] = self.__variance(sp.re(mat[k]))
                        else:
                            corr_r[k,j] = self.__correlation(sp.re(mat[k]), sp.re(mat[j]))
            
            for k in range(len(mat)):
                for j in range(len(mat)):
                    binary_1 = t in mat[k].free_symbols
                    binary_2 = t in mat[j].free_symbols
                    
                    if binary_1 == True and binary_2 == True:
                        if k == j:
                            corr_im[k,k] = self.__variance(sp.im(mat[k]))
                        else:
                            corr_im[k,j] = self.__correlation(sp.im(mat[k]), sp.im(mat[j]))
        
            corr_rs.append(corr_r)
            corr_ims.append(corr_im)

            self.__debugger('corr_rs',corr_rs, debug = 0)
            self.__debugger('corr_ims',corr_ims, debug = 0)

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
            
            result = s_r.reshape(len_resh, len_resh) + 1J*s_im.reshape(len_resh, len_resh)
            
            m += np.sqrt(self.gamma[ind])*result

        self.__debugger('stochpart',m, debug = 1)

        return(1J*m)
            
            
    def __modify_gate(self, U, det_part, corr_rs = None, corr_ims = None):
                                                   
        result = U @ scipy.linalg.expm(det_part) @ scipy.linalg.expm(self.__stoch_part(corr_rs, corr_ims))

        return result
    
    def twoqubit_single_run(self, psi_0, N, det, U, corr_mats):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        
        result = np.zeros([len(psi_0), N])
        
        result[:, 0] = psi_0
        
        U_array = np.array([self.__modify_gate(U, det_part = det, corr_rs = corr_mats[0], corr_ims = corr_mats[1]) for i in range(N)])
        for i in range(1,N):
            if i == 1:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))
                
                result[:, i] = np.diag(tmp)


                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))

                result[:, i] = np.diag(tmp)
                
        
        return(result)
    
    def singlequbit_single_run(self, psi_0, N, det, U, corr_mats):
        '''
        The actual run function.
        psi_0 > has to be for single qubit
        N >  number of single qubit gates.
        '''
        
        result = np.zeros([len(psi_0), N])
        
        result[:, 0] = psi_0        
        
        
        U_array = np.array([self.__modify_gate(U, det_part = det, corr_rs = corr_mats[0], corr_ims = corr_mats[1]) for i in range(N)])

        for i in range(1,N):
            if i == 1:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))

                result[:, i] = np.diag(tmp)
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                
                result[:, i] = np.diag(tmp)
                
        return(result)
    
    def twoqubit_sample_runs(self, psi_0, N, shots):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')

        H = self.two_qubit_gate_ryd_ham()
        U = scipy.linalg.expm(-1J*H)
        corr_mats = self.__stats(H)
        det = self.__det_part_r(H)

        with tqdm_joblib(tqdm(desc="My calculation", total=shots)) as progress_bar:
            res = Parallel(n_jobs=-1)(delayed(self.twoqubit_single_run)(psi_0, N, det, U, corr_mats) for i in range(shots))
        
        res = np.array(res).sum(axis=0)/(shots)
        
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        
        return(res)
    
    def singlequbit_sample_runs(self, psi_0, N, shots):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')

        H = self.single_qubit_gate_ryd_ham()
        U = scipy.linalg.expm(-1J*H)

        det = self.__det_part_r(H)
        corr_mats = self.__stats(H)

        with tqdm_joblib(tqdm(desc="My calculation", total=shots)) as progress_bar:
            res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N, det, U, corr_mats) for i in range(shots))
        
        #without progress bar: res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N) for i in range(shots))

        res = np.array(res).sum(axis=0)/(shots)
        
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        
        return(res)