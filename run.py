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
N = 3000
shots = 1
t1 = 0.04 #amplitude damping #4s is in thesis

t2 = 300*10**(-3) #dephasing damping

o_real = 2*np.pi*10*10**(3)


tg = np.pi/o_real

gamma_1 = tg/t1
gamma_2 = tg/t2

gamma = [gamma_1, gamma_1]

K_array = [np.array(([0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 1, 0, 0])), 
           np.array(([0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0], 
[1, 0, 0, 0]))]

'''dephasing

gamma = [gamma_2]

K_array = [np.array(([1, 0, 0, 0],
[0, -1, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0]))]
'''
o = np.pi
d = 0

tst = ng.rydberg_noisy_gate(K_array, o, d, 1, gamma)

results = tst.singlequbit_sample_runs(psi_0, N, shots)

plt.plot(results, color = 'tab:red', label = 'noisygate')
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
N = 3000
shots = 1

o = np.pi
d = 0.00001 #d = 0 causes overflow warnings, might want to add an exception
o_p = np.sqrt(o**2 + d**2)

results = tst.singlequbit_sample_runs(psi_0, N, shots, params = [1, o_p, d, o, 1, 1] ) #t, o_p, d, o, x1, x2
#plt.title('Time-evolution of |0> state')
plt.ylabel(r"$\rho_{00}$")
plt.xlabel(r'time in [$t_g$]')

#plt.axvline(x=t1/tg, label='$T_a$', color = 'orange', linestyle='dashed', alpha = 0.5)
#plt.axvline(x=t2/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)
plt.plot(results[0], color = 'tab:red', label = 'noisygate')

plt.legend()
#plt.savefig('noisygate_rydberg_sq_1000shots_10khzdrive_00.pdf', dpi=1000, bbox_inches = 'tight')
# %%
