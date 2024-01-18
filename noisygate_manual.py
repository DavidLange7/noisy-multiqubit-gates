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
from joblib import delayed, Parallel
import sympy as sp
from sympy.physics.quantum.dagger import Dagger as dgr

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
        
        U = scipy.linalg.expm(-1J*np.diag([1,1]))
        
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


        sample = np.random.multivariate_normal(np.array([0,0]), np.diag([1,1]), 1)
        st_arr = np.array([sample[0][i] for i in range(0,2)])
                
        #xi = 1J*eps_1*self.__sigm_gate()*st_arr[0] + 1J*eps_2*self.__sigp_gate()*st_arr[1]
        xi = 1J*np.array([[0,eps_2*st_arr[1]], [eps_1*st_arr[0],0]]) + 1J*eps_3*np.array([[1 + 1J*0, 0 + 1J*0], [0 + 1J*0, -1 + 1J*0]])
        
        return xi

    def __modify_gate_s(self, ev = 0):
        if ev == 1:
            U = self.single_qubit_gate()
            result = U @ scipy.linalg.expm(self.__det_part_s()) @ scipy.linalg.expm(self.__stoch_part_s())
            #result = U @ scipy.linalg.expm(self.__stoch_part_s())

        if ev == 0: 
            U = self.single_qubit_gate_rest()
            result = U @ scipy.linalg.expm(self.__det_part_1()) @ scipy.linalg.expm(self.__stoch_part_1())
            #result = U @ scipy.linalg.expm(self.__stoch_part_1())
        return result
    
    def singlegate_runs(self, psi_0, N, shots):
        '''
        The new run according to the code scheme ..
        
        This can be made better, later, now you give N number of single qubit gates and it computes the propagation
        
        The actual run function.
        psi_0 > has to be for single qubit
        N >  number of single qubit gates.
        '''
        
        results_st1 = np.zeros([N, shots])
        
        
        for j in range(0,shots):
            U_array = np.array([self.__modify_gate_s() for i in range(N)])
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
    
    def singlequbit_single_run(self, psi_0, N):
        '''
        The actual run function.
        psi_0 > has to be for two qubit
        N >  number of two qubit gates.
        '''
        
        results_p0 = np.zeros([N])
        results_p1 = np.zeros([N])


        tmp_st = []
        
        U_array = np.array([self.__modify_gate_s() for i in range(N)])
        for i in range(N):
            if i == 0:
                res = np.matmul(U_array[-i-1], psi_0)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])

                tmp_st.append(self.__modify_gate_s())
                
            else:
                res = np.matmul(U_array[-i-1], res)
                tmp = (np.outer(np.conj(res), res))
                results_p0[i] = np.real(tmp[0][0])
                results_p1[i] = np.real(tmp[1][1])

                tmp_st.append(self.__modify_gate_s())
        
        return(results_p0)
    
    def singlequbit_sample_runs(self, psi_0, N, shots):
        
        print('Start of simulation at ', datetime.datetime.now())
        print('--------------------------------')
        
        res = Parallel(n_jobs=-1)(delayed(self.singlequbit_single_run)(psi_0, N) for i in range(shots))
        
        res = np.array(res).sum(axis=0)/shots
        
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
tg = 35 * 10**(-9)

p = 0.0003216219528905892
T1_ctr = 0.0001389800498516866
T2_ctr = 0.00011220781847402393
T1_trg = 0.00015234783504581038
T2_trg = 8.365533281713897e-05
t1 = T1_ctr
t2 = T2_ctr
tg = 35 * 10**(-9)

tst = single_noisy_gate(np.pi,0, t1, t2,p)

psi_0 = np.zeros([2])
psi_0[1] = 1
N = 10000
shots = 1000
results_2 = tst.singlequbit_sample_runs(psi_0, N, shots)
#%%
#plt.title('Davids code')
plt.ylabel(r"$\rho_{00}$")
plt.xlabel(r'time in [t$_g$]')
plt.plot(results_2, label = 'noisygate', color = 'tab:red',  linewidth=3)

plt.axvline(t1/tg, color="forestgreen", label="$T_1$", ls="dashed")
plt.axvline(t2/tg, color="orange", label="$T_2$", ls="dashed")
plt.axvline(p/tg, color="grey", label="$T_d$")

plt.legend(fontsize = 45, loc='lower right')
plt.savefig('noisygate_rest_1000_1.pdf', dpi=1000, bbox_inches = 'tight')
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