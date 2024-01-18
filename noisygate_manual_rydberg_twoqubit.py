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
    def __init__(self, K_array, omega, V, delta = 0):
        '''
        Initialize run by giving the angles for single qubit gate
        '''
        self.omega = omega
        self.delta = delta
        self.V = V
        self.K_array = K_array
        
    def single_qubit_gate_ryd(self):
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        U = sp.Matrix(([sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) - 1J*d/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(-1J*o*x1/o_p*sp.sin(o_p/2*t)), 0, 0]
                          ,[sp.exp(1J*d*t/2)*(-1J*o*x2/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) + 1J*d/o_p*sp.sin(o_p/2*t)), 0, 0]
                          ,[0,0,1,0]
                          ,[0,0,0,1]))
        
        return U
            
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
        
    def __det_part_r(self, U):
                       
        lam = []
        for i in range(len(self.K_array)):
            L = np.matmul(np.matrix.conjugate(U), np.matmul(self.K_array[i], U))

            tmp = -1/2*(np.matmul(np.matrix.conjugate(L), L) - np.matmul(L, L))
            lam.append(tmp)
        lam_tot = np.sum(lam, axis = 0)

        return lam_tot
    
    def __stoch_part_r(self, U):
        
        xi = []
        
        for i in range(len(self.K_array)):
            L = np.matmul(np.matrix.conjugate(U), np.matmul(self.K_array[i], U))
                
            sample_i = np.array([np.random.multivariate_normal(np.zeros([16]), np.diag([1 for i in range(16)]), 1) for j in range(16)]).reshape([16,16])
            sample_r = np.array([np.random.multivariate_normal(np.zeros([16]), np.diag([1 for i in range(16)]), 1) for j in range(16)]).reshape([16,16])

            L = np.real(L)*sample_r + np.imag(L)*sample_i
                
            xi.append(L)
        
        xi_tot = np.sum(xi, axis = 0)

        return (1J*xi_tot)
    
    def __modify_gate_twoq(self, det_part = None, U_r = None):
                                                   
        U = U_r
        result = U @ scipy.linalg.expm(self.__stoch_part_r(U_r))
        #result = U @ scipy.linalg.expm(self.__stoch_part_r(K_array, params, st_mat, st_mat_deph))
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
        test = np.matmul(np.conjugate(U), np.matmul(self.K_array[0], U))
        print('testing')
        print(test.nonzero())


        det_part = self.__det_part_r(U)
        
        U_array = np.array([self.__modify_gate_twoq(det_part = det_part, U_r = U) for i in range(N)])
        for i in range(1,N):
            if i == 1:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_p5[i] = np.real(tmp[5][5])
                results_pd[i] = np.real(tmp[-1][-1])


                #tmp_st.append(self.__modify_gate_s(params = params, st_mat = st_mat, det_part = det_part))
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_p5[i] = np.real(tmp[5][5])
                results_pd[i] = np.real(tmp[-1][-1])


                #tmp_st.append(self.__modify_gate_s(params = params, st_mat = st_mat, det_part = det_part))
        
        return(results_p5)
    
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
'''
    def run(self, psi_0, shots):
        
        #the old try to do it, didnt quite work
        
        #The actual run function.
        #psi_0 > has to be for single qubit
        #shots >  number of stochastic runs you want to average over.
        
        dmat_n = np.zeros([2,2], dtype='complex128')
        shot_total = 0
        psi_new = 0
        
        for i in range(0,shots):
            U_new = self.__modify_gate_s()
            res = np.matmul(U_new, psi_0)
            
            res = res/(np.sqrt(np.abs(res[0])**2 + np.abs(res[1])**2))
            
            dmat = (np.outer(res, np.conj(res)))    
            dmat_n += dmat
            
            psi_new += res
            
            
            shot_result = np.square(np.absolute(res))
            shot_total += shot_result
            
        psi_new = psi_new/shots
        dmat_n = (dmat_n/shots)
        #print('density matrix of noisy gate:', dmat_n, 'and is its trace is', np.trace(dmat_n).real)

        #computing probabilities with density matrix
        tmp = np.array(([1,0,],[0,0]))
        probs = [np.trace(np.matmul(dmat_n, np.roll(tmp, i))) for i in [0,3]]

        #computing probabilities with statevector
        prob_n = shot_total/shots

        #plt.bar(['0', '1'], prob_n.flatten(), color ='blue')
        
        return(psi_new, dmat_n[0,0])


class twoqubit_noisy_gate():
    def __init__(self, theta, phi, t_gg, p, T1_ctr, T2_ctr, T1_trg, T2_trg):
        
        #Initialize run by giving the angles for single qubit gate
        
        
        if (T2 > 2*T1):
            raise Exception('wrong relaxation times given, make sure T2 <= 2*T1')
        self.theta = theta
        self.phi = phi
        self.T1 = T1
        self.T2 = T2
        self.p = p
        self.tg = 35 * 10**(-9)
        
    def 
'''
#%% half manual time evolution (just like the paper first all U noisy gates and then sampling over many shots)

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 20
shots = 2

o = 1
d = 0
V = 0.01
t1 = 1e5 #amplitude damping

gamma_1 = 1/t1

K_0_single = np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0]))

K_1_single = np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K_r_single = np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 1, 0]))

K_10_single = np.sqrt(gamma_1)*np.array(([0, 1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]))

K_r0_single = np.sqrt(gamma_1)*np.array(([0, 0, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]))

K_r1_single = np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]))

K = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
     np.kron(np.identity(4), K_1_single).reshape(16,16),
     np.kron(np.identity(4), K_r_single).reshape(16,16),
     np.kron(K_r_single, np.identity(4)).reshape(16,16),
     np.kron(K_0_single, np.identity(4)).reshape(16,16),
     np.kron(np.identity(4), K_0_single).reshape(16,16),
     np.kron(K_10_single, np.identity(4)).reshape(16,16),
     np.kron(np.identity(4), K_10_single).reshape(16,16),
     np.kron(K_r0_single, np.identity(4)).reshape(16,16),
     np.kron(np.identity(4), K_r0_single).reshape(16,16)]

#K = (np.array(K).sum(axis=0)).reshape(16,16)

tst = rydberg_twoq_noisy_gate(K, o, V, d)

results = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results, color = 'tab:red')
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)
