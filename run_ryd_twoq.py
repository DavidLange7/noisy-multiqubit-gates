#%%
import noisygate_rydberg_twoqubit as ng

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy

# half manual time evolution (just like the paper first all U noisy gates and then sampling over many shots)

psi_0 = np.zeros([16])
psi_0[5] = 1
N = 1200
shots = 1

o = 1
d = 0
V = 0.01
t1 = 1e2 #amplitude damping

gamma_1 = 1/t1


K_1_single = np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K = [np.kron(K_1_single, np.identity(4)).reshape(16,16)]

tst = ng.rydberg_twoq_noisy_gate(K, o, V, d)

results = tst.twoqubit_sample_runs(psi_0, N, shots)

plt.title('Time-evolution of |11> state')
plt.ylabel(r"$\rho_{11}$")
plt.xlabel(r'time')
plt.axvline(x=1e2, label='T1', color = 'orange', linestyle='dashed', alpha = 0.5)

plt.legend()
plt.plot(results, color = 'tab:red')
#plt.savefig('noisygatemanual_rydtwoq.pdf', dpi=1000)
#%%
#FIND THE ERROR
t = 1e2
gamma = 1/t
    
o = 1
d = 0
V = 0.01
t1 = 1e2 #amplitude damping

gamma_1 = 1/t1


K_1_single = np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K = [np.kron(K_1_single, np.identity(4)).reshape(16,16)]

tst = ng.rydberg_twoq_noisy_gate(K, o, V, d)

ham = tst.two_qubit_gate_ryd_ham()

val, vec = np.linalg.eig(-1J*ham)
v_m1 = (np.linalg.inv(vec))

t = sp.symbols('t', real = True)
r = [sp.exp(0.5*t), sp.exp(1.49751556253573*t), sp.exp(0.5*t), sp.exp(0.5*t), 1, sp.exp(0.997515562535733*t), 1, 1, sp.exp(0.5*t), 1, sp.exp(0.5*t), 1]
var = np.zeros([len(r)])
for i in range(len(r)):
    var[i] = float(sp.integrate(r[i]**2, (t, 0, 1)))
sample = np.random.multivariate_normal(np.zeros([len(r)]), np.diag(var), 1)
s = np.array([sample[0][i] for i in range(0,len(r))])

m = np.zeros([16, 16]) + 1J*np.zeros([16,16])

m[0][6] = 0.000886077110777928*s[0]
m[0][7] = -0.708862973703422*s[1]
m[0][8] = 0.5*s[2]
m[0][9] = -0.497506280743967*s[3]
m[1][6] = 0.705327387323585*s[4]
m[1][7] = 0.000881658076628822*s[5]
m[1][8] = 0.5*s[6]
m[1][9] = 0.502506218240452*s[7]

m[14][4] = 0.707106781186548*s[8]
m[14][5] = 0.707106781186547*s[9]
m[15][10] = -0.707106781186548*s[10]
m[15][11] = 0.707106781186547*s[11]

res = np.conj(v_m1) @ m @ v_m1
    
xi = np.sqrt(gamma)*res

lam = []
U = scipy.linalg.expm(-1J*ham)

for i in range(len(K)):
    L = np.matmul(np.conjugate(U), np.matmul(K[i], U))

    tmp = -1/2*(np.matmul(np.matrix.conjugate(L), L) - np.matmul(L, L))
    lam.append(tmp)
lam_tot = np.sum(lam, axis = 0)
# %%
