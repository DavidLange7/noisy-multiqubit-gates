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
#%%
class single_noisy_gate():
    def __init__(self, theta, phi, T1, T2, p):
        '''
        Initialize run by giving the angles for single qubit gate
        '''
        
        if (T2 > 2*T1):
            raise Exception('wrong relaxation times given, make sure T2 <= 2*T1')
        self.theta = theta
        self.phi = phi
        self.T1 = T1
        self.T2 = T2
        self.p = p
        self.tg = 35 * 10**(-9)

    def single_qubit_gate(self):
        ''' Returns general single qubit gate in noise free regime
        '''
        #X-gate > theta = np.pi
        theta = self.theta
        phi = self.phi
        
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        
        return U
    
    def single_qubit_gate_rest(self):
        
        U = np.diag([1,1])
        return U
    
    def single_qubit_gate_ryd(self):
        
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        U = sp.Matrix(([sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) - 1J*d/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(-1J*o*x1/o_p*sp.sin(o_p/2*t)), 0, 0]
                          ,[sp.exp(1J*d*t/2)*(-1J*o*x2/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) + 1J*d/o_p*sp.sin(o_p/2*t)), 0, 0]
                          ,[0,0,1,0]
                          ,[0,0,0,1]))
        
        return U
    
    def __get_lams(self, indx):
        '''
        Unpacks the wanted Lambda_k parameter
        0 > Lambda
        1 > Lambda_1
        2 > Lambda_2
        3 > Lambda_3
        etc.
        '''
        lam1 = self.p/4#self.p/4 #or 2*self.p# or self.p/2
        
        if self.T1 == 0:
            lam2 = 0#lam1
            lam3 = 0#lam1/2
        else:
            lam2 = self.tg/self.T1  + lam1
            lam3 = self.tg*(2*self.T1 - self.T2)/(4*self.T1*self.T2) + lam1/2
        
        if indx == 0:
            tmp = lam1 + lam2 + lam3
            
        if indx == 1:
            tmp = lam1
            
        if indx == 2:
            tmp = lam2
            
        if indx == 3:
            tmp = lam3
        
        if (indx != 0 and indx != 1 and indx != 2 and indx != 3):
            raise Exception("only integers 0,1,2,3 allowed") 
            
        return tmp
    
    def __sigp_gate(self):
        
        U = np.array([[0 + 1J*0, 1 + 1J*0], [0 + 1J*0, 0 + 1J*0]])
        return U
    
    def __sigm_gate(self):
        
        U = np.array([[0 + 1J*0, 0 + 1J*0], [1 + 1J*0, 0 + 1J*0]])
        return U
    
    def __Z_gate(self):
        
        U = np.array([[1 + 1J*0, 0 + 1J*0], [0 + 1J*0, -1 + 1J*0]])
        return U
    
    def __Y_gate(self):
        
        U = np.array([[0 + 1J*0, 0 + -1J*1], [0 + 1J*1, 0 + 1J*0]])
        return U
    
    def __X_gate(self):
        
        U = np.array([[0 + 1J*0, 1 + 1J*0], [1 + 1J*0, 0 + 1J*0]])
        return U
    
    def __identity(self):
        
        U = np.array([[1 + 1J*0, 0 + 1J*0], [0 + 1J*0, 1 + 1J*0]])
        return U
    
    def __R_xy(self, bar=False):
        
        if bar == False:
            phi = self.phi
        if bar == True:
            phi = self.phi + np.pi/2
            
        return(np.cos(phi)*self.__X_gate() + np.sin(phi)*self.__Y_gate())
    
    def __getstochmat_s(self):
        '''
        To get the covariant matrix for sampling the random variables.
        '''
        st_arr = np.zeros([8,8])
        
        for i in range(0,3):
            st_arr[i][i] = 1/2*(1+np.sin(2*self.theta)/(2*self.theta))
            
        for i in range(3,6):
            st_arr[i][i] = 1/2*(1-np.sin(2*self.theta)/(2*self.theta))
            
        for i in range(6,8):
            st_arr[i][i] = 1
            
        st_arr[2][5] = (1/(4*self.theta))*(1-np.cos(2*self.theta))
        st_arr[5][2] = st_arr[2][5]
        
        st_arr[0][3] = (1/(4*self.theta))*(1-np.cos(2*self.theta))
        st_arr[3][0] = st_arr[0][3]
        
        st_arr[1][4] = (1/(4*self.theta))*(1-np.cos(2*self.theta))
        st_arr[4][1] = st_arr[1][4]
        
        st_arr[0][6] = np.sin(self.theta)/self.theta
        st_arr[6][0] = st_arr[0][6]
        
        st_arr[1][7] = np.sin(self.theta)/self.theta
        st_arr[7][1] = st_arr[1][7]

        st_arr[3][6] = (1-np.cos(self.theta))/self.theta
        st_arr[6][3] = st_arr[3][6]
        
        st_arr[4][7] = (1-np.cos(self.theta))/self.theta
        st_arr[7][4] = st_arr[4][7]
        
        #print(np.isclose(st_arr,0).astype(int))
        #print(st_arr)
        return st_arr
    
    def __det_part_s(self):
        '''
        Deterministic part calculation.
        '''
        eps_1 = np.sqrt(self.__get_lams(1))
        eps_2 = np.sqrt(self.__get_lams(2))

        def R(theta):
            return(np.cos(theta/2)*self.__Z_gate() + np.sin(theta/2)*self.__R_xy(bar = True))
        
        det1 = (self.theta - np.sin(self.theta))/(2 * self.theta)
        det2 = (1-np.cos(self.theta))/self.theta
        det3 = (self.theta + np.sin(self.theta))/(2*self.theta)
        
        tmp_lam = -(eps_1**2 + eps_2**2)/4*self.__identity() - (eps_1**2 - eps_2**2)/4*np.sin(self.theta/2)/(self.theta/2)*R(self.theta)
        
        
        #e1 = np.sqrt(self.tg/self.T1)
        #tmp_lam = -e1**2/2 * np.array([[det1, 1J/2*np.exp(-1J*self.phi)*det2], [-1J/2*np.exp(1J*self.phi)*det2, det3]])
        return tmp_lam
    
    def __stoch_part_s(self):
        '''
        Stochastic part calculation.
        '''
        eps_1 = np.sqrt(self.__get_lams(1))
        eps_2 = np.sqrt(self.__get_lams(2))
        eps_3 = np.sqrt(self.__get_lams(3))
        
        st_mat = self.__getstochmat_s()
        
        sample = np.random.multivariate_normal(np.array([0,0,0,0,0,0,0,0]), st_mat, 1)
        st_arr = np.array([sample[0][i] for i in range(0,8)])
        
        #print(st_arr)
        
        f_0 = eps_3*st_arr[2] - 1J/2*(np.exp(1J*self.phi)*eps_2*st_arr[4] - np.exp(-1J*self.phi)*eps_1*st_arr[3])
        f_1 = 1/2*(np.exp(1J*self.phi)*eps_2*st_arr[7] + np.exp(-1J*self.phi)*eps_1*st_arr[6])
        f_2 = eps_3*st_arr[5] + 1J/2*(np.exp(1J*self.phi)*eps_2*st_arr[1] - np.exp(-1J*self.phi)*eps_1*st_arr[0])
        
        tmp_xi = 1J*f_0*self.__Z_gate() + 1J*f_1*self.__R_xy() + 1J*f_2*self.__R_xy(bar=True)
        
        return tmp_xi
    
    def __det_part_1(self):
        '''
        H = 1 evolution
        '''
        eps_1 = np.sqrt(self.__get_lams(1))
        eps_2 = np.sqrt(self.__get_lams(2))
        
        lam = -np.array([[eps_1**2/2,0], [0, eps_2**2/2]])
        
        return lam
    
    def __stoch_part_1(self):
        '''
        H = 1 evolution
        '''
        eps_1 = np.sqrt(self.__get_lams(1))
        eps_2 = np.sqrt(self.__get_lams(2))
        eps_3 = np.sqrt(self.__get_lams(3))
        
        sample = np.random.multivariate_normal(np.array([0,0,0]), np.diag([1,1,1]), 1)
        
        st_arr = np.array([sample[0][i] for i in range(0,3)])
        
        xi = 1J*eps_1*self.__sigm_gate()*st_arr[0] + 1J*eps_2*self.__sigp_gate()*st_arr[1] + 1J*eps_3*self.__Z_gate()*st_arr[2]
        return xi
        
    def __det_part_r(self, K_array, params):
               
        U = self.single_qubit_gate_ryd()
        
        lam = []
        o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
        x1, x2 = sp.symbols('x1, x2')
        
        for i in range(len(K_array)):
            L = (dgr(U)*K_array[i][0]*U)
            tmp1 = -1/2*(dgr(L)*L - L*L)
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
        
        gd = 0.00016666666666666666 #depolarization rate.
        g_1 = 1.2499999999999999e-05 #ampltiude rate.
        
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
        
        
        #val_1 = -(np.sqrt(1/gd)*o**2**st_arr[0])/(o_p**2) + np.sqrt(1/gd)*((d**2*st_arr[0]/(o_p**2)) + st_arr[1])
        #val_2 = -1*np.sqrt(1/gd)*o*x1*(-d*st_arr[0]/o_p + 1J* st_arr[2])/o_p - np.sqrt(1/gd)*o*x2*(-d*st_arr[0]/o_p + 1J*st_arr[2])/o_p
        #val_3 = 1*np.sqrt(1/gd)*o*x2*(d*st_arr[0]/o_p + 1J* st_arr[2])/o_p + np.sqrt(1/gd)*o*x1*(d*st_arr[0]/o_p + 1J*st_arr[2])/o_p
        #val_4 = (np.sqrt(1/gd)*o**2**st_arr[0])/(o_p**2) - np.sqrt(1/gd)*((d**2*st_arr[0]/(o_p**2)) + st_arr[1])
        
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
    
    def __modify_gate_s(self, ev = 2, params = [1,1,1,1,1,1], K_array = None, st_mat = None, st_mat_deph = None, det_part = None, U_r = None):
                                                   
        if ev == 1:
            U = self.single_qubit_gate()
            result = U @ scipy.linalg.expm(self.__det_part_s()) @ scipy.linalg.expm(self.__stoch_part_s())
        #result = U @ scipy.linalg.expm(self.__stoch_part_s())

        if ev == 0:
            U = self.single_qubit_gate_rest()
            result = U @ scipy.linalg.expm(self.__det_part_1()) @ scipy.linalg.expm(self.__stoch_part_1())
            
        if ev == 2:
            U = U_r
            params = params[1:]
            result = U @ scipy.linalg.expm(det_part)@ scipy.linalg.expm(self.__stoch_part_r(K_array, params, st_mat, st_mat_deph))
            #result = U @ scipy.linalg.expm(self.__stoch_part_r(K_array, params, st_mat, st_mat_deph))    
        return result
    '''-also you can use dephasing error introduced by noises in laser using a Z-lindblad-term ..
    '''
    
    def singlegate_runs(self, psi_0, N, shots, params, K_array = [
            [sp.sqrt(1/1e2)*sp.Matrix(([0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [1, 0, 0, 0]))], 
            [sp.sqrt(1/1e2)*sp.Matrix(([0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0]))] ]):
        '''
        The new run according to the code scheme ..
        
        This can be made better, later, now you give N number of single qubit gates and it computes the propagation
        
        The actual run function.
        psi_0 > has to be for single qubit
        N >  number of single qubit gates.
        '''
        
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
                    
                if i == int(N/2) and j == int(shots/2):
                    print('Halfwaypoint of simulation reached at ', datetime.datetime.now())
            
        
        results_st2 = np.array([np.sum(results_st1[k])/shots for k in range(N)])
        plt.plot(results_st2)
        
        return(results_st2)
    
    def singlequbit_single_run(self, psi_0, N, params, K_array = [ 
            [sp.sqrt(1.2499999999999999e-05)*sp.Matrix(([0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [1, 0, 0, 0]))], [sp.sqrt(1.2499999999999999e-05)*sp.Matrix(([0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 1, 0, 0]))] ]):
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
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_pd[i] = np.real(tmp[3][3])

                #tmp_st.append(self.__modify_gate_s(params = params, st_mat = st_mat, det_part = det_part))
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])
                
                results_pd[i] = np.real(tmp[3][3])

                #tmp_st.append(self.__modify_gate_s(params = params, st_mat = st_mat, det_part = det_part))
        
        return(results_p0, results_pd)
    
    def singlequbit_sample_runs(self, psi_0, N, shots, params):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        with tqdm_joblib(tqdm(desc="My calculation", total=shots)) as progress_bar:
            res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N, params) for i in range(shots))
        
        #res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N, params) for i in range(shots))
        
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
tst = single_noisy_gate(np.pi,0, 0.00016645649920534426, 5.1104092647143704e-05, 0.00032169487086029)
psi_0 = np.zeros([4])
psi_0[0] = 1
N = 50000
shots = 100
t1 = 4 #amplitude damping
t2 = 300*10**(-3) #depol damping

o_real = 2*np.pi*10*10**(3)
o_fin = np.pi

tg = np.pi/o_real

gamma_1 = tg/t1
gamma_2 = tg/t2

o = np.pi
d = 0
o_p = np.sqrt(o**2 + d**2)
results = tst.singlequbit_sample_runs(psi_0, N, shots, params = [1, o_p, d, o, 1, 1] ) #t, o_p, d, o, x1, x2
#%%
#plt.title('Time-evolution of |0> state')
plt.ylabel(r"$\rho_{00}$")
plt.xlabel(r'time in [$t_g$]')
o_fin = o
tg = np.pi/o_real
t1 = 4 #amplitude damping
t2 = 300*10**(-3) #depol damping
gamma_1 = tg/t1
gd = tg/t2

#plt.axvline(x=t1/tg, label='$T_a$', color = 'orange', linestyle='dashed', alpha = 0.5)
plt.axvline(x=t2/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)
plt.plot(results[0], color = 'tab:red', label = 'noisygate')

plt.legend()
plt.savefig('noisygate_rydberg_sq_1000shots_10khzdrive_00.pdf', dpi=1000, bbox_inches = 'tight')

#%%
plt.title('Davids code')
plt.ylabel(r"$\rho_{00}$")
plt.xlabel(r'time in [t$_g$ units]')
plt.plot(results)

plt.axvline(x=0.00032169487086029/(35 * 10**(-9)), label='p', color = 'blue', linestyle='dashed', alpha = 0.5)
plt.axvline(x=0.00016645649920534426/(35 * 10**(-9)), label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)
plt.axvline(x=5.1104092647143704e-05/(35 * 10**(-9)), label='T2', color = 'red', linestyle='dashed', alpha = 0.5)
plt.legend()
#plt.savefig('noisygatemanual.pdf', dpi=1000)

#%% TIME EVOLUTION

N = 1000
psi_0 = np.zeros([2])
psi_0[0] = 1
''' 
FROM IBM MANILA DEVICE PARAMETERS
{'T1': [8.27711395727174e-05,
  0.00017636357549110463,
  9.8870017570622e-05,
  0.0002245686161730805,
  0.0001104632705526117],
 'T2': [7.249960222840705e-05,
  5.6324438981865633e-05,
  2.1265514457652728e-05,
  6.158967405761706e-05,
  4.053624214111765e-05],
 'p': [0.0005635030527165414,
  0.00026462322348501713,
  0.00020921016684936463,
  0.0001788748344194181,
  0.0003039888182763828]}
 
 0.00016769455815350716, 5.85174071477909e-05, 0.0020947596750693505)
'''

tst = single_noisy_gate(np.pi,0, 0.00016210914184504208, 0.00010947158425524756, 0.0003329764080022075)
results_dmat = []
results_psi = []    
results_psi.append(psi_0)

for i in range(0,N):
    tmp = tst.run(psi_0,1000)
    psi_0 = tmp[0]#/(np.sqrt(np.abs(tmp[0][0])**2 + np.abs(tmp[0][1])**2))
    results_dmat.append(tmp[1])
    results_psi.append(psi_0)