#%%
import os
#os.chdir('/home/david/Courses_padova/Thesis/publishment/noisy-multiqubit-gates/')

import noisygate_rydberg_numerical as ng
import noisy_rydberg_singlequbit_analytic as nrsa
import csv
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy
import seaborn as sns
#%%

t1 = 0.2 #amplitude damping #4s is in thesis
o_real = 2*np.pi*10*10**(3)
tg = np.pi/o_real

gamma_1 = tg/t1
omega = np.pi
delta = 0
x1 = 1
x2 = 1
V = 0

K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

init = ng.rydberg_noisy_gate([K_1_single], omega, delta, V, x1, x2, [gamma_1])
params = omega, delta, V, x1, x2
x_gate = init.gate_only(params, 1, 1)

params = omega/2, delta, V, -1J,1J
ysqrt_gate = init.gate_only(params, 1, 1)
hadamard = x_gate @ ysqrt_gate


#%%
def Cz_gate(control, target, plot = True):

    eps = 1.894
    omega = 5*2*np.pi
    V = 3000
    delta = omega/eps
    t1 = 14.99
    tau = t1*2*np.pi/np.sqrt(omega**2 + delta**2)
    t1 = 400
    gamma_1 = tau/t1
    K = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_1_single).reshape(16,16)]
    gamma = [gamma_1 for i in range(len(K))]

    params = [omega*tau, delta*tau, V*tau, 1, 1]
    init = ng.rydberg_noisy_gate(K, omega, delta, V, x1, x2, gamma)
    cz = init.gate_only(params, 2, 1)
    
    basis_labels = ['00', '01', '0r', '0d', 
                   '10', '11', '1r', '1d', 
                   'r0', 'r1', 'rr', 'rd', 
                   'd0', 'd1', 'dr', 'dd']
    
    
    if plot == True:
        plt.figure('real')
        sns.heatmap(np.real(cz), annot = True, fmt=".2f", cmap="YlGnBu",
                    yticklabels=basis_labels, 
                    xticklabels=basis_labels, 
        annot_kws={"size": 15},  # Font size for annotations
        cbar_kws={"shrink": 0.8} # Adjust colorbar size (optional)
                    
        )
        plt.yticks(fontsize=14, rotation=0)
        plt.xticks(fontsize=14, rotation=0)
        plt.title("cz_operator matrix", fontsize = 14)
    
    return cz
#%%
'''
In this cell we run the single and two qubit rydberg noisy gate using the numerical version
> diagonalization is done numerically
calling noisygate_rydberg_numerical

'''
# half manual time evolution (just like the paper first all U noisy gates and then sampling over many shots)
#Single qubit gate
results_num_arr = []
for shots in [5]:
       psi_0 = np.zeros([4])
       psi_0[0] = 1
       N = 5
       t1 = 0.2 #amplitude damping #4s is in thesis

       t2 = 300*10**(-3) #dephasing damping

       o_real = 2*np.pi*10*10**(3)


       tg = np.pi/o_real

       gamma_1 = tg/t1
       gamma_2 = tg/t2

       gamma = [gamma_1, gamma_1, gamma_2]

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

       tst = ng.rydberg_noisy_gate(K_array, o, d, 1, 1, 1, gamma)

       results_num = tst.singlequbit_sample_runs(psi_0, N, shots)

       with open(f'res_{shots}_nov22_500_sq.txt', 'w') as f:
              csv.writer(f, delimiter=',').writerows(results_num)
       results_num_arr.append(results_num)
#%%

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 1000
shots = 5000
o = np.pi
d = 0
V = 1
t1 = 50*10**(-6) #amplitude damping

tg = 0.5*10**(-6)

gamma_1 = tg/t1

K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_1_single).reshape(16,16)]
gamma = [gamma_1 for i in range(len(K))]

tst = ng.rydberg_noisy_gate(K, o, d, V, gamma)


results_1 = tst.twoqubit_sample_runs(psi_0, N, shots)
with open('dec6_v1_shots5000_newstoch.txt', 'w') as f:
    csv.writer(f, delimiter=',').writerows(results_1)

#plt.plot(results_num, color = 'tab:red', label = 'noisygate')
#plt.axvline(x=t1/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)
#%%
'''
In this cell, we run the single qubit rydberg noisy gate in its analytical form,
calling noisy_rydberg_singlequbit_analytic

#half manual time evolution (just like the paper first all U noisy gates and then sampling over many shots)

'''
results_anly_arr = []

for shots in [10, 50, 200, 500]:

       K_array = [
        [sp.Matrix(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0]))], 
        [sp.Matrix(([0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0]))],
                   sp.Matrix(([1, 0, 0, 0],
       [0, -1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]))]
        
       tst = nrsa.single_noisy_gate(np.pi, 0, K_array, gamma)

       psi_0 = np.zeros([4])
       psi_0[0] = 1
       N = 7000

       o = np.pi
       d = 1e-10 #d = 0 causes overflow warnings, might want to add an exception
       o_p = np.sqrt(o**2 + d**2)

       results_anly = tst.singlequbit_sample_runs(psi_0, N, shots, params = [1, o_p, d, o, 1, 1] ) #t, o_p, d, o, x1, x2
       results_anly_arr.append(results_anly)

#plt.title('Time-evolution of |0> state')
#plt.ylabel(r"$\rho_{00}$")
#plt.xlabel(r'time in [$t_g$]')

#plt.axvline(x=t1/tg, label='$T_a$', color = 'orange', linestyle='dashed', alpha = 0.5)
#plt.axvline(x=t2/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)
#plt.plot(results_anly[0], color = 'tab:red', label = 'noisygate')

#plt.legend()
#plt.savefig('noisygate_rydberg_sq_1000shots_10khzdrive_00.pdf', dpi=1000, bbox_inches = 'tight')
# %%
plt.plot(np.abs(results_num - results_anly))

#%%
#two qubit case

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 500
shots = 1000
o = np.pi
d = 0
V = 3
t1 = 50*10**(-6) #amplitude damping

tg = 0.5*10**(-6)

gamma_1 = tg/t1

K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))
K_0_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0]))

K_array = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_1_single).reshape(16,16),
           np.kron(K_0_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_0_single).reshape(16,16)]



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
K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_1_single).reshape(16,16)]
gamma = [gamma_1 for i in range(len(K))]

tst = ng.rydberg_noisy_gate(K, o, d, V, gamma)


results_3 = tst.twoqubit_sample_runs(psi_0, N, 500)
with open('res_500_nov16.txt', 'w') as f:
    csv.writer(f, delimiter=',').writerows(results_3)
tst = ng.rydberg_noisy_gate(K, o, d, V, gamma)


results_1 = tst.twoqubit_sample_runs(psi_0, N, 1000)
with open('res_1000_nov16.txt', 'w') as f:
    csv.writer(f, delimiter=',').writerows(results_1)
tst = ng.rydberg_noisy_gate(K, o, d, V, gamma)


results_2 = tst.twoqubit_sample_runs(psi_0, N, 2000)
with open('res_2000_nov16.txt', 'w') as f:
    csv.writer(f, delimiter=',').writerows(results_2)


plt.title('Time-evolution of |11> state')
#plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results_4[1], color = 'tab:red')
'''
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)
with open('res_jun22.txt', 'w') as f:
    csv.writer(f, delimiter=',').writerows(results_1)
'''
#two qubit case
#%%
psi_0 = np.zeros([16])
psi_0[5] = 1
N = 800
shots = 500


tst = ng.rydberg_noisy_gate(K_array, o, d, V, gamma)


results_2 = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
#plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results_2[1], color = 'tab:red')

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 800
shots = 1000


tst = ng.rydberg_noisy_gate(K_array, o, d, V, gamma)


results_3 = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
#plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results_3[1], color = 'tab:red')
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)
#%%
#two qubit case

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 1500
shots = 30

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


gamma = [gamma_1 for i in range(len(K_array))]


tst = ng.rydberg_noisy_gate(K_array, o, d, V, gamma)


results_3 = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
#plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results_3[1], color = 'tab:red')
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 1500
shots = 350

o = np.pi
d = 0
V = 5
t1 = 50*10**(-6) #amplitude damping

tg = 0.5*10**(-6)

gamma_1 = tg/t1

K_1_single = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K_array = [np.kron(K_1_single, np.identity(4)).reshape(16,16),
           np.kron(np.identity(4), K_1_single).reshape(16,16)]


gamma = [gamma_1 for i in range(len(K_array))]


tst = ng.rydberg_noisy_gate(K_array, o, d, V, gamma)


results_3 = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
#plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results_3[1], color = 'tab:red')
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)
# %%