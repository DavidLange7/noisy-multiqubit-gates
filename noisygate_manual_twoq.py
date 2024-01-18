#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:34:04 2023

@author: david
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime
from joblib import delayed, Parallel

import pylab as plb

plb.rcParams['font.size'] = 40
plt.rcParams["figure.figsize"] = (18,12)
#%%
class twoqubit_noisy_gate():
    def __init__(self, theta, phi, t_gg, p, T1_ctr, T2_ctr, T1_trg, T2_trg):
        '''
        Initialize run by giving the angles for single qubit gate
        '''
        
        if (T2_ctr > 2*T1_ctr):
            raise Exception('wrong relaxation times given, make sure T2 <= 2*T1 on control qubit')
        if (T2_trg > 2*T1_trg):
            raise Exception('wrong relaxation times given, make sure T2 <= 2*T1 on target qubit')
            
        self.theta = theta
        self.phi = phi
        self.T1_ctr = T1_ctr
        self.T2_ctr = T2_ctr
        self.T1_trg = T1_trg
        self.T2_trg = T2_trg
        self.p = p
        self.tg = 35 * 10**(-9)

    def single_qubit_gate(self):
        '''
        Returns general single qubit gate in noise free regime
        '''
        #X-gate > theta = np.pi
        theta = self.theta
        phi = self.phi
        
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U
    
    def two_qubit_gate_rest(self):
        
        U = np.diag([1,1,1,1])
        return U
    
    def two_qubit_gate(self):
        '''
        Returns general two qubit gate in noise free regime
        '''
        U = np.array(
            [[np.cos(self.theta/2), -1J*np.sin(self.theta/2) * np.exp(-1J * self.phi), 0, 0],
             [-1J*np.sin(self.theta/2) * np.exp(1J * self.phi), np.cos(self.theta/2), 0, 0],
             [0, 0, np.cos(self.theta/2), 1J*np.sin(self.theta/2) * np.exp(-1J * self.phi)],
             [0, 0, 1J*np.sin(self.theta/2) * np.exp(1J * self.phi), np.cos(self.theta/2)]]
        )
        return U
    
    def __get_lams(self, indx, qu_index):
        '''
        Unpacks the wanted Lambda_k parameter
        0 > Lambda
        1 > Lambda_1
        etc.
        '''
        if (qu_index != 1 and qu_index != 2):
            raise ValueError("Enter qu_index = 1 or 2 to select one of the two qubits")
        
        lam1 = 2*self.p/4#self.p/4 #or 2*self.p# or self.p/2
        
        
        if qu_index == 1:
            
            if self.T1_ctr == 0:
                lam2 = 0#lam1
                lam3 = 0#lam1/2
            else:
                lam2 = self.tg/self.T1_ctr  + lam1
                lam3 = self.tg*(2*self.T1_ctr - self.T2_ctr)/(4*self.T1_ctr*self.T2_ctr) + lam1/2
            
            if indx == 0 and qu_index == 1:
                tmp = lam1 + lam2 + lam3
                
            if indx == 1:
                tmp = lam1
                
            if indx == 2:
                tmp = lam2
                
            if indx == 3:
                tmp = lam3
            
            if (indx != 0 and indx != 1 and indx != 2 and indx != 3):
                raise Exception("only integers 0,1,2,3 allowed")
                
        if qu_index == 2:
            
            if self.T1_trg == 0:
                lam2 = 0#lam1
                lam3 = 0#lam1/2
            else:
                lam2 = self.tg/self.T1_trg  + lam1
                lam3 = self.tg*(2*self.T1_trg - self.T2_trg)/(4*self.T1_trg*self.T2_trg) + lam1/2
            
            if indx == 0 and qu_index == 1:
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
    
    def __sigm_p(self):
        
        U = 1/2*(np.array([[0 + 1J*0, 1 + 1J*0], [1 + 1J*0, 0 + 1J*0]]) + 1J*np.array([[0 + 1J*0, 0 + -1J*1], [0 + 1J*1, 0 + 1J*0]]))
        return U

    def __sigm_m(self):
        
        U = 1/2*(np.array([[0 + 1J*0, 1 + 1J*0], [1 + 1J*0, 0 + 1J*0]]) - 1J*np.array([[0 + 1J*0, 0 + -1J*1], [0 + 1J*1, 0 + 1J*0]]))
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
    
    def __getstochmat(self):
        '''
        To get the covariant matrix for sampling the random variables.
        '''
        st_arr = np.zeros([9,9])
        
        for i in range(0,3):
            st_arr[i][i] = 1/2*(1+np.sin(2*self.theta)/(2*self.theta))
            
        for i in range(3,6):
            st_arr[i][i] = 1/2*(1-np.sin(2*self.theta)/(2*self.theta))
            
        for i in range(6,9):
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
        
        st_arr[2][8] = np.sin(self.theta)/self.theta
        st_arr[8][2] = st_arr[2][8]
        
        st_arr[5][8] = (1-np.cos(self.theta))/self.theta
        st_arr[8][5] = st_arr[5][8]
        
        #print(np.isclose(st_arr,0).astype(int))
        #print(st_arr)
        return st_arr
    
    def __det_part_d(self):
        '''
        Deterministic part calculation.
        '''
        eps_1_ctr = np.sqrt(self.__get_lams(1,1))
        eps_2_ctr = np.sqrt(self.__get_lams(2,1))
        
        eps_1_trg = np.sqrt(self.__get_lams(1,2))
        eps_2_trg = np.sqrt(self.__get_lams(2,2))

        def Z_times_R(theta):
            return(np.cos(theta/2)*np.kron(self.__Z_gate(), self.__Z_gate()) + np.sin(theta/2)*np.kron(self.__identity(), self.__R_xy(bar = True)))
        
        tmp_lam_test1 = -1/2* np.array(
              [[eps_1_ctr**2, 0, 0, 0],
               [0, eps_2_ctr**2, 0, 0],
               [0, 0, eps_1_trg**2, 0],
               [0, 0, 0, eps_2_trg**2]])
        
        tmp_lam_test2 = -1/2* np.array(
              [[eps_1_trg**2+eps_1_ctr**2, 0, 0, 0],
               [0, eps_2_ctr**2 + eps_2_trg**2, 0, 0],
               [0, 0, eps_1_trg**2 + eps_1_ctr**2, 0],
               [0, 0, 0, eps_2_ctr**2 + eps_2_trg**2]])
        
        tmp_lam_1 = -(eps_1_ctr**2 + eps_2_ctr**2)/2 * np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])
        tmp_lam_2 = -(eps_1_trg**2 + eps_2_trg**2)/2 * np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
        tmp_lam_3 = - (eps_1_trg**2 - eps_2_trg**2)/4*(np.kron(self.__Z_gate(), self.__identity())\
                    + np.sin(self.theta/2)/(self.theta/2)*(np.cos(self.theta/2)*np.kron(self.__identity(), self.__Z_gate()) + np.sin(self.theta/2)*Z_times_R(self.theta)))\
        
        tmp_lam = tmp_lam_test2 + tmp_lam_3
        
        #Deterministic contribution given by relaxation
        det1 = (self.theta-np.sin(self.theta))/(2*self.theta)
        det2 = (self.theta)*(1-np.cos(self.theta))
        det3 = (2*self.theta)*(self.theta+np.sin(self.theta))

        deterministic_r_ctr = -eps_1_ctr**2/2 * np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
        
        deterministic_r_trg = -eps_1_trg**2/2 * np.array(
            [[det1,1J*(1/2)*det2*np.exp(-1J*self.phi),0,0],
             [-1J*(1/2)*det2*np.exp(1J*self.phi),det3,0,0],
             [0,0,det1,-1J*(1/2)*det2*np.exp(-1J*self.phi)],
             [0,0,1J*(1/2)*det2*np.exp(1J*self.phi),det3]])
        
        return tmp_lam#deterministic_r_trg+deterministic_r_ctr
    
    def __stoch_part_d(self):
        '''
        Stochastic part calculation.
        '''
        eps_1_ctr = np.sqrt(self.__get_lams(1,1))
        eps_2_ctr = np.sqrt(self.__get_lams(2,1))
        eps_3_ctr = np.sqrt(self.__get_lams(3,1))
        
        eps_1_trg = np.sqrt(self.__get_lams(1,2))
        eps_2_trg = np.sqrt(self.__get_lams(2,2))
        eps_3_trg = np.sqrt(self.__get_lams(3,2))

        st_mat_1 = self.__getstochmat()
        sample_1 = np.random.multivariate_normal(np.array([0,0,0,0,0,0,0,0,0]), st_mat_1, 1)
        st_arr_1 = np.array([sample_1[0][i] for i in range(0,9)])
        
        sample_2 = np.random.multivariate_normal(np.array([0,0,0,0,0,0,0,0,0]), st_mat_1, 1)
        st_arr_2 = np.array([sample_2[0][i] for i in range(0,9)])
        
        xi_1 = eps_3_ctr*st_arr_1[8]*np.kron(self.__Z_gate(), self.__identity()) + eps_1_ctr*st_arr_1[0]*np.kron(self.__sigm_m(), self.__identity())\
             + eps_1_ctr*1J*st_arr_1[3]*np.kron(self.__sigm_m(), self.__R_xy(bar = False)) + eps_2_ctr*st_arr_1[1]*np.kron(self.__sigm_p(), self.__identity())\
             + eps_2_ctr*st_arr_1[4]*np.kron(self.__sigm_p(), self.__R_xy(bar = False))
             #eps_3_trg*st_arr_1[8]*np.kron(self.__Z_gate, self.__identity()) + 
        tmp = st_arr_1[1]
        f_w = 1/2*(eps_1_trg*np.exp(-1J*self.phi)*st_arr_2[6] + eps_2_trg*np.exp(1J*self.phi)*st_arr_2[7])
        f_p = 1/2*(eps_1_trg*np.exp(-1J*self.phi)*st_arr_2[0] - eps_2_trg*np.exp(1J*self.phi)*st_arr_2[1])
        f_m = 1/2*(eps_1_trg*np.exp(-1J*self.phi)*st_arr_2[3] - eps_2_trg*np.exp(1J*self.phi)*st_arr_2[4])

        xi_2 = 1J*f_w*np.kron(self.__identity(), self.__R_xy(bar = False)) - f_m*np.kron(self.__Z_gate(), self.__Z_gate()) + f_p*np.kron(self.__identity(), self.__R_xy(bar = True))\
             + 1J*eps_3_trg*st_arr_2[2]*np.kron(self.__identity(), self.__Z_gate()) + 1J*eps_3_trg*st_arr_2[5]*np.kron(self.__Z_gate(), self.__R_xy(bar=True))
             
        
        return [1J*(xi_1) + xi_2, tmp]
    
    def modify_gate_d(self, ev = 1):
        
        if ev == 0:
            U = self.two_qubit_gate_rest()
            result = U @ scipy.linalg.expm(self.__det_part_d()) @ scipy.linalg.expm(self.__stoch_part_d()[0])
            tmp = self.__stoch_part_d()[1]
            
        if ev == 1:
            U = self.two_qubit_gate()
            result = U @ scipy.linalg.expm(self.__det_part_d()) @ scipy.linalg.expm(self.__stoch_part_d()[0])
            tmp = self.__stoch_part_d()[1]
            
        return [result, tmp]

    def twoqubit_runs(self, psi_0, N, shots):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        
        results_p10 = np.zeros([N, shots])
        results_p00 = np.zeros([N, shots])
        results_p01 = np.zeros([N, shots])

        tmp_st = []
        
        for j in range(0,shots):
            U_array = np.array([self.modify_gate_d(ev=1)[0] for i in range(N)])
            for i in range(N):
                if i == 0:
                    res = np.matmul(U_array[-i-1], psi_0)
                    tmp = (np.outer(np.conj(res), res))
                    results_p10[i][j] = np.real(tmp[2][2])
                    results_p00[i][j] = np.real(tmp[0][0])
                    results_p10[i][j] = np.real(tmp[1][1])


                    tmp_st.append(self.modify_gate_d(ev=1)[1])
                else:
                    res = np.matmul(U_array[-i-1], res)
                    tmp = (np.outer(np.conj(res), res))
                    results_p10[i][j] = np.real(tmp[2][2])
                    results_p00[i][j] = np.real(tmp[0][0])
                    results_p01[i][j] = np.real(tmp[1][1])

                    tmp_st.append(self.modify_gate_d(ev=1)[1])
                    
                if i == int(N/2) and j == int(shots/2):
                    print('Halfwaypoint of simulation reached at ', datetime.datetime.now())
            
        
        results_p10 = np.array([np.sum(results_p10[k])/shots for k in range(N)])
        results_p00 = np.array([np.sum(results_p00[k])/shots for k in range(N)])
        results_p01 = np.array([np.sum(results_p01[k])/shots for k in range(N)])

        plt.plot(results_p10)
        
        plt.figure('2')
        plt.hist(tmp_st, 100)
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        return(results_p00, results_p01, results_p10)
    

    def twoqubit_single_run(self, psi_0, N):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        
        results_p10 = np.zeros([N])
        results_p00 = np.zeros([N])
        results_p01 = np.zeros([N])
        results_p11 = np.zeros([N])
        
        tmp_st = []
        
        U_array = np.array([self.modify_gate_d(ev=1)[0] for i in range(N)])
        for i in range(N):
            if i == 0:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))
                results_p10[i] = np.real(tmp[2][2])
                results_p00[i] = np.real(tmp[0][0])
                results_p01[i] = np.real(tmp[1][1])

                tmp_st.append(self.modify_gate_d(ev=1)[0])
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                results_p10[i] = np.real(tmp[2][2])
                results_p00[i] = np.real(tmp[0][0])
                results_p01[i] = np.real(tmp[1][1])

                tmp_st.append(self.modify_gate_d(ev=1)[0])
        
        return(results_p00, results_p01, results_p10)
    
    def twoqubit_sample_runs(self, psi_0, N, shots):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        
        res = Parallel(n_jobs=-1)(delayed(self.twoqubit_single_run)(psi_0, N) for i in range(shots))
        
        res = np.array(res).sum(axis=0)/shots
        
        print('--------------------------------')
        print('End of simulation at ', datetime.datetime.now())
        
        return(res)

#p = 0.00030318431494484
#T1 = 0.0001575547667160676
#T2= 9.621808627732993e-05

tg = 35 * 10**(-9)

p = 0.0003216219528905892
T1_ctr = 0.0001389800498516866
T2_ctr = 0.00011220781847402393
T1_trg = 0.00015234783504581038
T2_trg = 8.365533281713897e-05