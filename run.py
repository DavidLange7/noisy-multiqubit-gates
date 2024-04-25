#%%
import os
os.chdir('/home/david/Courses_padova/Thesis/publishment/noisy-multiqubit-gates/')

import noisygate_rydberg_numerical as ng
import noisy_rydberg_singlequbit_analytic as nrsa

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy
#%%
'''
In this cell we run the single and two qubit rydberg noisy gate using the numerical version
> diagonalization is done numerically
calling noisygate_rydberg_numerical

'''
# half manual time evolution (just like the paper first all U noisy gates and then sampling over many shots)
#Single qubit gate

psi_0 = np.zeros([4])
psi_0[0] = 1
N = 400
shots = 2
t1 = 0.04 #amplitude damping #4s is in thesis

t2 = 300*10**(-3) #dephasing damping

o_real = 2*np.pi*10*10**(3)


tg = np.pi/o_real

gamma_1 = tg/t1
gamma_2 = tg/t2

gamma = [gamma_1, gamma_1, gamma_1]

K_array = [np.array(([0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 1, 0, 0])), 
           np.array(([0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0], 
[1, 0, 0, 0])),
            np.array(([1, 0, 0, 0],
[0, -1, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0]))]

o = np.pi
d = 0

tst = ng.rydberg_noisy_gate(K_array, o, d, 1, gamma)

results_num = tst.singlequbit_sample_runs(psi_0, N, shots)

plt.plot(results_num, color = 'tab:red', label = 'noisygate')
#plt.axvline(x=t1/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)
# %%
'''
In this cell, we run the single qubit rydberg noisy gate in its analytical form,
calling noisy_rydberg_singlequbit_analytic

#half manual time evolution (just like the paper first all U noisy gates and then sampling over many shots)

'''


t1 = 0.04 #amplitude damping
t2 = 300*10**(-3) #dephasing

o_real = 2*np.pi*10*10**(3)
o_fin = np.pi

tg = np.pi/o_real

g_1 = tg/t1
gd = tg/t2

gamma = [g_1, g_1]

K_array = [
        [sp.Matrix(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0]))], 
        [sp.Matrix(([0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0]))] ]
        
tst = nrsa.single_noisy_gate(np.pi, 0, K_array, gamma)

psi_0 = np.zeros([4])
psi_0[0] = 1
N = 2000
shots = 100

o = np.pi
d = 0.00001 #d = 0 causes overflow warnings, might want to add an exception
o_p = np.sqrt(o**2 + d**2)

results_anly = tst.singlequbit_sample_runs(psi_0, N, shots, params = [1, o_p, d, o, 1, 1] ) #t, o_p, d, o, x1, x2
#plt.title('Time-evolution of |0> state')
plt.ylabel(r"$\rho_{00}$")
plt.xlabel(r'time in [$t_g$]')

#plt.axvline(x=t1/tg, label='$T_a$', color = 'orange', linestyle='dashed', alpha = 0.5)
#plt.axvline(x=t2/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)
plt.plot(results_anly[0], color = 'tab:red', label = 'noisygate')

plt.legend()
#plt.savefig('noisygate_rydberg_sq_1000shots_10khzdrive_00.pdf', dpi=1000, bbox_inches = 'tight')
# %%
plt.plot(np.abs(results_num - results_anly))

#%%
#two qubit case

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 300
shots = 1

o = np.pi
d = 0
V = 0.01
t1 = 50*10**(-6) #amplitude damping

tg = 0.5*10**(-6)

gamma_1 = tg/t1

K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K_array = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_1_single).reshape(16,16)]



K_0_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0]))

K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K_r_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 1, 0]))

K_10_single = np.array(([0, 1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]))

K_r0_single = np.array(([0, 0, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]))

K_r1_single = np.array(([0, 0, 0, 0],
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

gamma = [gamma_1 for i in range(len(K_array))]


tst = ng.rydberg_noisy_gate(K_array, o, d, V, gamma)


results_1 = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
#plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results_1[1], color = 'tab:red')
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)
# %%
