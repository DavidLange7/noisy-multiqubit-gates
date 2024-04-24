"""
Qutip example
=============

This examples shows how to run and reproduce the experiment of
the preprint arxiv:2301.04173v2 using qutip, by solving the
linbland equation.  
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qp
import sympy as sp
from sympy.physics.quantum.dagger import Dagger as dgr
from qutip import tensor, basis
import scipy
import pylab as plb

plb.rcParams['font.size'] = 45
plt.rcParams["figure.figsize"] = (18,12)


# If you want to define an operator that is not
# from the ones given by qutip (or a given initial state)
# do not worry: you just need to create your operator/state
# of interest in numpy and then call the qp.Qobj() class, just like:
# my_op_in_numpy = np.array([ [1, 0], [0, 1] ]) 
# my_op = qp.Qobj(my_op_in_numpy)
def single_gate_hamiltonian(theta, phi):
    """
    Define the single-qubit gate hamiltonian
    from the paper arxiv:2301.04173v2

    Parameters
    ----------
    theta : float
        Multiplicative parameter of the hamiltonian
    phi : float
        phase of the Rx in the hamiltonian
    """

    hamiltonian = np.cos(phi)*qp.sigmax() + np.sin(phi)*qp.sigmay()
    hamiltonian *= theta/2

    return hamiltonian

def single_gate_hamiltonian_rest():
    """
    Define the single-qubit gate hamiltonian
    from the paper arxiv:2301.04173v2

    Parameters
    ----------
    theta : float
        Multiplicative parameter of the hamiltonian
    phi : float
        phase of the Rx in the hamiltonian
    """

    hamiltonian = np.diag([1,1])
    hamiltonian = qp.Qobj(hamiltonian, dims = [[2],[2]])

    return hamiltonian

def single_gate_hamiltonian_ryderg(Delta, Omega, x_1 = 1, x_2 = 1):
    hamiltonian = np.array([[0, Omega/2*x_1], [Omega/2*x_2, -Delta]])
    hamiltonian = qp.Qobj(hamiltonian)
    
    return hamiltonian

def single_gate_hamiltonian_ryderg_fourlevels(Delta, Omega, x_1 = 1, x_2 = 1):
    hamiltonian = np.array([[0, Omega/2*x_1, 0, 0], [Omega/2*x_2, -Delta, 0, 0], [0,0,1,0], [0,0,0,1]])
    hamiltonian = qp.Qobj(hamiltonian)
    
    return qp.Qobj(hamiltonian.data.toarray().reshape((4,4)),
            dims=[[2,2],[2,2]])

def two_qubit_gate_rydberg(delta, omega, x_1, x_2, V):
    '''
    resulting order: 00, 01, 0r, 10, 11, 1r, r0, r1, rr
    '''
    sigp = tensor(qp.Qobj([[0,0,1]]).dag(), qp.Qobj([[0,1,0]]))
    sigm = tensor(qp.Qobj([[0,1,0]]).dag(), qp.Qobj([[0,0,1]]))
    n = tensor(qp.Qobj([[0,0,1]]).dag(), qp.Qobj([[0,0,1]]))

    H_1 = omega/2*(sigm + sigp) - delta*n
    H_2 = qp.tensor(H_1, qp.identity(3)).data + qp.tensor(qp.identity(3), H_1).data + V*qp.Qobj(qp.tensor(n, qp.identity(3)).data* qp.tensor(qp.identity(3), n).data)
    H_2 = qp.Qobj(H_2)

    return H_2


def two_qubit_gate_rydberg_w_dark(delta, omega, x_1, x_2, V):
    '''
    resulting order: 00, 01, 0r, 0d, 10, 11, 1r, 1d, r0, r1, rr, rd, d0, d1, dr, dd
    '''
    sigp = tensor(qp.Qobj([[0,0,1,0]]).dag(), qp.Qobj([[0,1,0,0]]))
    sigm = tensor(qp.Qobj([[0,1,0,0]]).dag(), qp.Qobj([[0,0,1,0]]))
    n = tensor(qp.Qobj([[0,0,1,0]]).dag(), qp.Qobj([[0,0,1,0]]))

    H_1 = omega/2*(sigm + sigp) - delta*n
    H_2 = qp.tensor(H_1, qp.identity(4)).data + qp.tensor(qp.identity(4), H_1).data + V*qp.Qobj(qp.tensor(n, qp.identity(4)).data* qp.tensor(qp.identity(4), n).data)
    H_2 = qp.Qobj(H_2)
    
    return H_2

def two_qubit_gate(theta, phi):
    '''
    Returns general two qubit gate in noise free regime
    '''
    hamiltonian = qp.tensor(qp.sigmaz(), np.cos(phi)*qp.sigmax() + np.sin(phi)*qp.sigmay())
    hamiltonian *= theta/2

    return hamiltonian
#%%
'''
Single gate superconducting
'''
# Time discretization
initial_time = 0.0
final_time = 15000.0        # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*1
times = np.linspace(initial_time, final_time, num_timesteps)

# Define hamiltonian for X gate
theta = np.pi
phi = 0.0 # Not sure here, it is left variable in the cern code (?)
hamiltonian = single_gate_hamiltonian(np.pi, phi)

# Define the limbland operators. Here you should put the correct parameters
tg = 35 * 10**(-9)

p = 0.0003216219528905892
T1_ctr = 0.0001389800498516866
T2_ctr = 0.00011220781847402393
T1_trg = 0.00015234783504581038
T2_trg = 8.365533281713897e-05

t1 = T1_ctr
t2 = T2_ctr
tg = 35 * 10**(-9)

gamma_d = p
gamma_1 = tg/t1
gamma_z = tg*(2*t1 - t2)/(4*t1*t2)

ll1 = gamma_d/4
ll2 = gamma_d/4 + gamma_1
ll3 = gamma_d/8 + gamma_z

#stuff for rest hamiltonian 
#U = scipy.linalg.expm(-1J*np.diag([1,1]))
#hamiltonian = single_gate_hamiltonian_rest()

print(ll2/(ll1+ll2))

U = scipy.linalg.expm(-1J*hamiltonian)

linb1 = np.sqrt(ll1)*np.matmul(np.conjugate(U), np.matmul(np.array(qp.sigmam()), U))
linb2 = np.sqrt(ll2)*np.matmul(np.conjugate(U), np.matmul(np.array(qp.sigmap()), U))

linb3 = np.sqrt(ll3)*np.matmul(np.conjugate(U), np.matmul(np.array(qp.sigmaz()), U))
############################################

#linb1 = np.sqrt(ll1)*(np.exp(-1J*phi)/2 *((np.cos(phi)*qp.sigmax() + np.sin(phi)*qp.sigmay()) - 1J*(np.cos(theta + np.pi/2)*qp.sigmaz() + np.sin(theta + np.pi/2)*(np.cos(phi+np.pi/2)*qp.sigmax() + np.sin(phi+np.pi/2)*qp.sigmay()))))
#linb2 = np.sqrt(ll2)*(np.exp(1J*phi)/2 *((np.cos(phi)*qp.sigmax() + np.sin(phi)*qp.sigmay()) + 1J*(np.cos(theta + np.pi/2)*qp.sigmaz() + np.sin(theta + np.pi/2)*(np.cos(phi+np.pi/2)*qp.sigmax() + np.sin(phi+np.pi/2)*qp.sigmay()))))
#linb3 = np.sqrt(ll3)*(np.cos(theta)*qp.sigmaz() + np.sin(theta)*(np.cos(phi+np.pi/2)*qp.sigmax() + np.sin(phi+np.pi/2)*qp.sigmay()))

linb1 = qp.Qobj(linb1, dims = [[2], [2]])
linb2 = qp.Qobj(linb2, dims = [[2], [2]])
linb3 = qp.Qobj(linb3, dims = [[2], [2]])

#linb3 = qp.Qobj([[+0.01154143+0.00000000e+00j,  0 + 0j], [ 0+0J,  -0.01154143+0.00000000e+00j]])

linblands = [linb3]


# Initial state as |0>
# The first number is the number of state in the basis (2 for qubits)
# The second number is the populated state (we start from the GS |0>)
# To treat multiple qubits qp.fock([2, 2], [0, 0])
psi0 = qp.fock(2, 1)

result = qp.mesolve(
                    H     = hamiltonian,
                    rho0  = psi0,
                    tlist = times,
                    c_ops = linblands,        # Linbland operators
                    e_ops = [qp.Qobj([[1,0],[0,0]]), qp.sigmay(), qp.sigmax()]     # Expectation values operators
                    )


# Put to true if you want just to watch the evolution after the application of gates
full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0

# Map sigma_z into qubits
pop2 = (result.expect[0])
pop3 = (result.expect[1])
pop4 = (result.expect[2])


ax.plot(times[mask], pop3[mask], label="qutip", linewidth = 2.5)
#plt.plot(tmp, label = 'noisygate', color = 'tab:red',  linewidth=2.5)
#ax.plot(result_3, label="noisygate", color = 'tab:red', linewidth=1.5)

plt.xlim([-5,10000])
#ax.axvline(t1/tg, color="forestgreen", label="T1", ls="dashed")
#ax.axvline(t2/tg, color="orange", label="T2", ls="dashed")
#ax.axvline(p/tg, color="grey", label="Td")
ax.set_xlabel('time in [$t_g$]')
ax.set_ylabel(r"$\rho_{00}$")
ax.axvline(T1_ctr/tg, color="forestgreen", label="$T_1 = 138.98\: \mathrm{\mu s}$", ls="dashed")
ax.axvline(T2_ctr/tg, color="orange", label="$T_2 = 112.21\: \mathrm{\mu s}$", ls="dashed")
ax.axvline(p/tg, color="grey", label="$T_d = 321.62\: \mathrm{\mu s}$")
#plt.axvline(t1/tg, color="forestgreen", label="$T_1$", ls="dashed")
#plt.axvline(t2/tg, color="orange", label="$T_2$", ls="dashed")
#plt.axvline(p/tg, color="grey", label="$T_d$")
ax.legend(fontsize = 45, loc='lower right')
#plt.savefig('noisygate_rest_5000_dec10.png', dpi=500, bbox_inches = 'tight')
#%%
fig = plt.figure()
ax = fig.add_subplot()

line, = ax.plot(np.abs(tmp[10:]-pop2[10:10000]), color='green', lw=2)
plt.xlabel('Times in [$t_g$]')
plt.ylabel(r"$|\rho_{00,exact} - \rho_{00,ng}|$")

ax.set_yscale('log')
ax.set_xscale('log')
plt.grid()
plt.savefig('noisygate_rest_5000_dec10_unnorm.png', dpi=500, bbox_inches = 'tight')
#%%
'''
two-qubit superconducting CR gate
'''
# Time discretization
initial_time = 0.0
final_time = 10000.0        # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*1
times = np.linspace(initial_time, final_time, num_timesteps)


# Define hamiltonian for CR gate
theta = np.pi
phi = 0.0
hamiltonian = two_qubit_gate(theta, phi)

# Define the lindbland operators.
tg = 35 * 10**(-9)


p = 0.0003216219528905892
t1_ctr = 0.0001389800498516866
t2_ctr = 0.00011220781847402393

t1_trg = 0.00015234783504581038
t2_trg = 8.365533281713897e-05

mat1 = qp.Qobj([[0,0],[0,1]])

mat2 = qp.Qobj([[1,0],[0,0]])


gamma_d = p/4 # Depolarizing probability

gamma_1_ctr = tg/t1_ctr
gamma_z_ctr = tg*(2*t1_ctr - t2_ctr)/(4*t1_ctr*t2_ctr)

gamma_1_trg = tg/t1_trg
gamma_z_trg = tg*(2*t1_trg - t2_trg)/(4*t1_trg*t2_trg)

ll1 = 2*gamma_d
ll2 = 2*gamma_d + gamma_1_ctr
ll3 = gamma_d + gamma_z_ctr

ll4 = 2*gamma_d
ll5 = 2*gamma_d + gamma_1_trg
ll6 = gamma_d + gamma_z_trg


#interaction picture?
linb1 = np.sqrt(ll1)*qp.tensor(qp.sigmam(), (-1J*theta*qp.sigmax()).expm())
linb2 = np.sqrt(ll2)*qp.tensor(qp.sigmap(), (1J*theta*qp.sigmax()).expm())
linb3 = np.sqrt(ll3)*qp.tensor(mat2, qp.sigmaz())

linb4 = np.sqrt(ll4)*(-1J*hamiltonian).expm()

linb4 = np.sqrt(ll4)*(np.exp(-1J*phi)/2 * ((qp.tensor(mat1, np.cos(phi)*qp.sigmax() + np.sin(phi)*qp.sigmay())) - 1J*(np.cos(theta+np.pi/2)*qp.tensor(qp.sigmaz(), qp.sigmaz())) + np.sin(theta+np.pi/2)*qp.tensor(qp.identity(2), np.cos(phi+np.pi/2)*qp.sigmax() + np.sin(phi+np.pi/2)*qp.sigmay())))
linb5 = np.sqrt(ll5)*(np.exp(1J*phi)/2 * ((qp.tensor(mat1, np.cos(phi)*qp.sigmax() + np.sin(phi)*qp.sigmay())) + 1J*(np.cos(theta+np.pi/2)*qp.tensor(qp.sigmaz(), qp.sigmaz())) + np.sin(theta+np.pi/2)*qp.tensor(qp.identity(2), np.cos(phi+np.pi/2)*qp.sigmax() + np.sin(phi+np.pi/2)*qp.sigmay())))
linb6 = np.sqrt(ll6)*((np.cos(theta)*qp.tensor(mat1, qp.sigmaz())) + np.sin(theta)*qp.tensor(qp.sigmaz(), np.cos(phi+np.pi/2)*qp.sigmax() + np.sin(phi+np.pi/2)*qp.sigmay()))

lindblands = [linb1, linb2, linb3, linb4, linb5, linb6]


psi0 = qp.fock([2, 2], [1, 1])

result = qp.mesolve(
                    H     = hamiltonian,
                    rho0  = psi0,
                    tlist = times,
                    c_ops = lindblands,        # Linbland operators
                    e_ops = [qp.tensor(qp.Qobj([[0,0],[0,1]]), qp.Qobj([[0,0],[0,1]]))]    # Expectation values operators
                    )


# Put to true if you want just to watch the evolution after the application of gates
full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0

# Map sigma_z into qubits
pop1 = result.expect[0]

ax.plot(times[mask], pop1[mask], label="qutip")
#ax.plot(res_4[2], label="noisygate", color = 'tab:red', alpha = 0.5)
ax.axvline(T1_ctr/tg, color="forestgreen", label="$T_{1,trg}$", ls="dashed")
ax.axvline(T2_ctr/tg, color="orange", label="$T_{2,trg}$", ls="dashed")
ax.axvline(p/tg, color="grey", label="$T_d$")
ax.set_xlabel('time in [$t_g$]')
ax.set_ylabel(r"$\rho_{22}$")
ax.legend(loc='upper right')
#plt.savefig('asdfboth_scq_twogate_2000.pdf', dpi=1000, bbox_inches='tight')
plt.show()
#%%
'''
This section is for exact solution of Rydberg gates, simple two-level rabi oscillation with ampli damping
expected behaviour: drive to the fully mixed state (0.5 expectation value for both levels)
'''
# Time discretization
initial_time = 0.0
final_time = 1000.0        # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*10
times = np.linspace(initial_time, final_time, num_timesteps)

# Define hamiltonian for X gate
d = 0
o = 1
o_p = np.sqrt(d**2 + o**2)

x_1 = 1
x_2 = 1

hamiltonian = single_gate_hamiltonian_ryderg(d, o)

# Define the limbland operators. Here you should put the correct parameters
t1 = 1e2 #amplitude damping

gamma_1 = 1/t1

ll1 = gamma_1
linb1 = np.sqrt(ll1)*qp.Qobj(np.array([[1J*o/o_p*x_2*np.sin(o_p/2)*(np.cos(o_p/2) - 1J*d/o_p*np.sin(o_p/2)), o**2/(o_p**2)*x_1*x_2*np.sin(o_p/2)*np.sin(o_p/2)], 
                                       [(np.cos(o_p/2) - 1J*d/o_p*np.sin(o_p/2))**2, -1J*o/o_p*x_1*np.sin(o_p/2)*(np.cos(o_p/2) - 1J*d/o_p*np.sin(o_p/2))]]))

lindblands = [linb1]

psi0 = qp.fock(2, 0)

result = qp.mesolve(
                    H     = hamiltonian,
                    rho0  = psi0,
                    tlist = times,
                    c_ops = lindblands,        # Linbland operators
                    e_ops = [qp.sigmaz()]     # Expectation values operators
                    )

# Put to true if you want just to watch the evolution after the application of gates
full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0

# Map sigma_z into qubits
pop1 = (result.expect[0]+1)/2

ax.plot(times[mask], pop1[mask], label="$\\langle\\sigma_z\\rangle$")
ax.axvline(t1, color="forestgreen", label="T1", ls="dashed")
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.legend()
plt.show()
#%%
'''
Lets try with the operators and 4 level system 0,1,r,d
'''
# Time discretization
initial_time = 0.0
final_time = 1000     # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*2000
times = np.linspace(initial_time, final_time, num_timesteps)

initial_time = 0.0
final_time = 50000.0 # Set a shorter final time for faster simulation
num_timesteps = 70000
times = np.linspace(initial_time, final_time, num_timesteps)


o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
x1, x2 = sp.symbols('x1, x2')


# Define the lindblad operators. Here you should put the correct parameters
t1 = 4 #amplitude damping
t2 = 300*10**(-3) #p.deph damping

o_real = 2*np.pi*10*10**(3)
o_fin = np.pi

tg = np.pi/o_real

gamma_1 = tg/t1
gamma_2 = tg/t2


ll1 = gamma_1
U = sp.Matrix(([sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) - 1J*d/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(-1J*o*x1/o_p*sp.sin(o_p/2*t)), 0, 0]
                  ,[sp.exp(1J*d*t/2)*(-1J*o*x2/o_p*sp.sin(o_p/2*t)), sp.exp(1J*d*t/2)*(sp.cos(o_p/2*t) + 1J*d/o_p*sp.sin(o_p/2*t)), 0, 0]
                  ,[0,0,1,0]
                  ,[0,0,0,1]))


K_1 = sp.sqrt(gam1)*sp.Matrix(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0]))

K_2 = sp.sqrt(gam1)*sp.Matrix(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))

K_d = sp.sqrt(gam2)*sp.Matrix(([1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]))
'''
K_2 = sp.Matrix(([1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, sp.sqrt(1-gam1)]))
'''

L_1 = dgr(U)*K_1*U
L_2 = dgr(U)*K_2*U

U_dag = sp.lambdify([o_p, t, d, o, x1, x2, gam1], dgr(U), "numpy")
U = sp.lambdify([o_p, t, d, o, x1, x2, gam1], U, "numpy")
K_d = sp.lambdify([o_p, t, d, o, x1, x2, gam2], K_d, "numpy")

numpy_expr = sp.lambdify([o_p, t, d, o, x1, x2, gam1], L_1, "numpy")
numpy_expr2 = sp.lambdify([o_p, t, d, o, x1, x2, gam1], L_2, "numpy")


d = 0
o = o_fin
o_p = np.sqrt(d**2 + o**2)

x_1 = 1
x_2 = 1

hamiltonian = single_gate_hamiltonian_ryderg_fourlevels(d, o)

numpy_expr3 = U_dag(o_p, 1, d, o, x_1, x_2, gamma_1)@K_d(o_p, 1, d, o, x_1, x_2, gamma_2)@U(o_p, 1, d, o, x_1, x_2, gamma_1)


linb1 = qp.Qobj(numpy_expr(o_p, 1, d, o, x_1, x_2, gamma_1).reshape((4,4)),
        dims=[[2,2],[2,2]])

linb2 = qp.Qobj(numpy_expr2(o_p, 1, d, o, x_1, x_2, gamma_1).reshape((4,4)),
        dims=[[2,2],[2,2]])

linbd = qp.Qobj(numpy_expr3.reshape((4,4)),
        dims=[[2,2],[2,2]])


lindblands = [linbd]

psi0 = qp.fock([2, 2], [0, 0])

result = qp.mesolve(
                    H     = hamiltonian,
                    rho0  = psi0,
                    tlist = times,
                    c_ops = lindblands,
                    e_ops = [qp.tensor(qp.Qobj([[1,0],[0,0]]), qp.Qobj([[1,0],[0,0]])), qp.tensor(qp.Qobj([[0,0],[0,1]]), qp.Qobj([[0,0],[0,1]])), qp.tensor(qp.Qobj([[1,0],[0,0]]), qp.Qobj([[0,1],[0,0]])), qp.tensor(qp.Qobj([[1,0],[0,0]]), qp.Qobj([[0,0],[1,0]]))]     # Expectation values operators
                    )

# Put to true if you want just to watch the evolution after the application of gates
full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0
pop1 = result.expect[0]
pop2 = result.expect[1]

#%%
# Map sigma_z into qubits
fig, ax = plt.subplots()
plt.axvline(x=t2/tg, label='$T_{dp}$', color = 'black', linestyle='dashed', alpha = 0.7)

ax.plot(times, pop3, label = 'qutip')
#ax.plot(results[1],  label="noisygate", color = 'tab:red', alpha = 1)
#ax.plot(results[1], label="noisygate", color = 'tab:red', alpha = 1)

#ax.plot(times[mask], np.abs(pop1[mask] - results[0][:500])/(pop1[mask]), label = 'qutip')

#ax.axvline(t1, color="forestgreen", label="$T_a$", ls="dashed")

ax.set_xlabel('time in [$t_g$]')
ax.set_ylabel('$\\rho_{dd}$')

#plt.ylim([0,1])
#ax.set_ylabel('$|\\rho_{00,qtp} - \\rho_{00,ng}|/\\rho_{00,qtp}$')
#plt.xlim([-5,500])
ax.legend()
plt.show()
plt.savefig('both_rydberg_sq_1000shots_10khzdrive_dd.pdf', dpi=1000, bbox_inches='tight')

#%%
'''
Lets try two qubit rydberg and use 4 level system 0,1,r,d
'''
# Time discretization
initial_time = 0.0
final_time = 400 # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*1
times = np.linspace(initial_time, final_time, num_timesteps)

o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
x1, x2 = sp.symbols('x1, x2')

# Define the limbland operators. Here, you should put the correct parameters
t1 = 50*10**(-6) #amplitude damping

tg = 0.5*10**(-6)

gamma_1 = tg/t1

d = 0
o = np.pi
x1 = 1
x2 = 1
V = 0.01

#hamiltonian = two_qubit_gate_rydberg(d, o, 1, 1, 100)
hamiltonian = two_qubit_gate_rydberg_w_dark(d, o, 1, 1, V)

val_o, vec = np.linalg.eig(-1J*hamiltonian)
val = np.exp(val_o)


val_mat = np.diag(val)


#U = np.matmul(vec, np.matmul(val_mat,np.linalg.inv(vec)))
U = scipy.linalg.expm(-1J*hamiltonian)
# FOR the expectation values:
tmp = np.zeros([16,16])
tmp2 = np.zeros([16,16])
tmp3 = np.zeros([16,16])
tmp4 = np.zeros([16,16])
tmp5 = np.zeros([16,16])
tmp6 = np.zeros([16,16])


i = 5
i2 = -2
i3 = -4
i4 = -1
i5 = 7
i6 = -3

tmp[i,i] = 1
tmp2[i2,i2] = 1
tmp3[i3,i3] = 1
tmp4[i4,i4] = 1
tmp5[i5,i5] = 1
tmp6[i6,i6] = 1


###########################

ll1 = gamma_1
'''
resulting order: 00, 01, 0r, 10, 11, 1r, r0, r1, rr

resulting order: 00, 01, 0r, 0d, 10, 11, 1r, 1d, r0, r1, rr, rd, d0, d1, dr, dd
'''
'''
amplitude damping: 0d>dd (3>-1), 1d>dd (7>-1), rd>dd (11>-1) and also 12,13,14 > 15
'''

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

#we can try with one only, so the sampling in the noisy gate is easier..
K = [np.kron(K_1_single, qp.identity(4)).reshape(16,16),
     np.kron(qp.identity(4), K_1_single).reshape(16,16)]
'''
     np.kron(qp.identity(4), K_1_single).reshape(16,16),
     np.kron(qp.identity(4), K_r_single).reshape(16,16),
     np.kron(K_r_single, qp.identity(4)).reshape(16,16),
     np.kron(K_0_single, qp.identity(4)).reshape(16,16),
     np.kron(qp.identity(4), K_0_single).reshape(16,16),
     np.kron(K_10_single, qp.identity(4)).reshape(16,16),
     np.kron(qp.identity(4), K_10_single).reshape(16,16),
     np.kron(K_r0_single, qp.identity(4)).reshape(16,16),
     np.kron(qp.identity(4), K_r0_single).reshape(16,16),
     np.kron(K_r1_single, qp.identity(4)).reshape(16,16),
     np.kron(qp.identity(4), K_r1_single).reshape(16,16)]
'''

#test = np.conj(np.linalg.inv(vec)) @ val_mat @ np.conj(vec) @ K[0] @ vec @ np.conj(val_mat) @ np.linalg.inv(vec)

linb = [qp.Qobj(np.matmul(np.conjugate(U), np.matmul(K[j], U)),
        dims=[[4,4],[4,4]]) for j in range(len(K))]
#linb = [qp.Qobj(np.linalg.multi_dot([np.conjugate(np.linalg.multi_dot([vec, val_mat, np.linalg.inv(vec)])), K[i], vec, val_mat, np.linalg.inv(vec)]), dims=[[4,4],[4,4]]) for i in range(len(K))]
        
'''
linb1 = qp.Qobj(K_1,
        dims=[[4,4],[4,4]])
linb2 = qp.Qobj(K_2,
        dims=[[4,4],[4,4]])
linb3 = qp.Qobj(K_3,
        dims=[[4,4],[4,4]])
'''

hamiltonian = qp.Qobj(hamiltonian, dims=[[4,4],[4,4]])

lindblands = [linb]

#question: why is the probability from 0 to .25 for starting in |11> of |1r> and |r1> ?

#psi0 = qp.fock([3, 3], [1, 1])
psi0 = qp.fock([4, 4], [1, 1])

result = qp.mesolve(
                    H     = hamiltonian,
                    rho0  = psi0,
                    tlist = times,
                    c_ops = lindblands,
                    e_ops = [qp.Qobj(qp.Qobj(tmp).data.toarray().reshape(16,16), dims = [[4,4], [4,4]]), 
                             qp.Qobj(qp.Qobj(tmp2).data.toarray().reshape(16,16), dims = [[4,4], [4,4]]), 
                             qp.Qobj(qp.Qobj(tmp3).data.toarray().reshape(16,16), dims = [[4,4], [4,4]]),
                             qp.Qobj(qp.Qobj(tmp4).data.toarray().reshape(16,16), dims = [[4,4], [4,4]]),
                             qp.Qobj(qp.Qobj(tmp5).data.toarray().reshape(16,16), dims = [[4,4], [4,4]]),
                             qp.Qobj(qp.Qobj(tmp6).data.toarray().reshape(16,16), dims = [[4,4], [4,4]])]    # Expectation values operators
                    )
# Put to true if you want just to watch the evolution after the application of gates
full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0

# Map sigma_z into qubits
pop1 = result.expect[0]
pop2 = result.expect[1]
pop3 = result.expect[2]
pop4 = result.expect[3]
pop5 = result.expect[4]
pop6 = result.expect[5]


U_k = ((np.fft.fft(pop1)))

ax.plot(times[mask], pop1[mask], label= r"$\langle \mathrm{11} \rangle$")
ax.plot(times[mask], pop2[mask] + pop3[mask] + pop5[mask] + pop6[mask], label= r"$\langle \mathrm{d1} \rangle + \langle \mathrm{1d} \rangle+ \langle \mathrm{rd}\rangle + \langle \mathrm{dr} \rangle$")
ax.plot(times[mask], pop4[mask], label= r"$\langle \mathrm{dd} \rangle$")

ax.axvline(t1/tg, color="black", label="$T_a$", ls="dashed")


#ax.axvline(t1, color="forestgreen", label="T1", ls="dashed")
ax.set_xlabel('Time [$t_g$]')
ax.set_ylabel('Expectation values')
ax.legend(fontsize = 35)
#plt.savefig('qutip_twoqubit.pdf', dpi=1000, bbox_inches='tight')

plt.show()
#%%
'''
Here I try the final part: so two qubit with 4 levels 0,1,r,d so it will be a 9x9 without dark and 
16x16 with the dark state: 
    resulting order: 00, 01, 0r, 10, 11, 1r, r0, r1, rr
    resulting order: 00, 01, 0r, 0d, 10, 11, 1r, 1d, r0, r1, rr, rd, d0, d1, dr, dd
    
exponentiating meaning eigenvalue>> not feasable with 9x9 or more ofc. So solution?
Maybe we can decompose the interacting part (0 and dark states are not involved in any dynamics only relaxation.)
'''
from sympy.physics.quantum import TensorProduct

def two_qubit_gate_rydberg_1r(delta, omega, x_1, x_2, V):
    '''
    resulting order: 00, 01, 0r, 10, 11, 1r, r0, r1, rr
    '''
    sigp = tensor(qp.Qobj([[0,1]]).dag(), qp.Qobj([[1,0]]))
    sigm = tensor(qp.Qobj([[1,0]]).dag(), qp.Qobj([[0,1]]))
    n = tensor(qp.Qobj([[0,1]]).dag(), qp.Qobj([[0,1]]))

    H_1 = omega/2*(sigm + sigp) - delta*n
    H_2 = qp.tensor(H_1, qp.identity(2)).data + qp.tensor(qp.identity(2), H_1).data + V*qp.Qobj(qp.tensor(n, qp.identity(2)).data* qp.tensor(qp.identity(2), n).data)
    H_2 = qp.Qobj(H_2)

    return H_2

def two_qubit_gate_rydberg_1r_symb():
    '''
    resulting order: 00, 01, 0r, 10, 11, 1r, r0, r1, rr
    '''
    
    o_p, t, d, o, gam1, gam2, V = sp.symbols('o_p, t, d, o, gam1, gam2, V', real = True)
    x1, x2 = sp.symbols('x1, x2')
    
    sigp = tensor(qp.Qobj([[0,1]]).dag(), qp.Qobj([[1,0]]))
    sigm = tensor(qp.Qobj([[1,0]]).dag(), qp.Qobj([[0,1]]))
    n = tensor(qp.Qobj([[0,1]]).dag(), qp.Qobj([[0,1]]))

    o = 1
    V = 1
    H_1 = o/2*(sp.Matrix(sigm) + sp.Matrix(sigp)) #- d*sp.Matrix(n)
    n = sp.Matrix(n)
    H_2 = TensorProduct(H_1, sp.eye(2)) + TensorProduct(sp.eye(2), H_1) + V*TensorProduct(n, sp.eye(2))*TensorProduct(sp.eye(2), n)

    return H_2

ham = two_qubit_gate_rydberg_1r_symb()

initial_time = 0.0
final_time = 500.0        # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*100
times = np.linspace(initial_time, final_time, num_timesteps)

delta = 0
omega = 1
V = 1
#%%
hamiltonian = two_qubit_gate_rydberg_1r(delta, omega, 1, 1, V)
psi0 = qp.fock([2, 2], [0, 0])
result = qp.mesolve(
                    H     = hamiltonian,
                    rho0  = psi0,
                    tlist = times,
                    e_ops = [qp.tensor(qp.Qobj([[0,0],[0,1]]), qp.Qobj([[0,0],[0,1]]))]     # Expectation values operators
                    )
#Put to true if you want just to watch the evolution after the application of gates
full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0

# Map sigma_z into qubits
pop1 = result.expect[0]

U_k = np.fft.fftshift(np.fft.fft(pop1))

ax.plot(times[mask], pop1[mask], label= r"$\langle \mathrm{rr} \rangle$")
#ax.axvline(t1, color="forestgreen", label="T1", ls="dashed")
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.legend()
plt.show()
#%% 
# This is for steady state solution (rest gate sperconducting single qubit..)
p_00, p_01, p_10, p_11 = sp.symbols('p_00, p_10, p_01, p_11')
f, gamma_1, gamma_2, gamma_3 = sp.symbols('f, gamma_1, gamma_2, gamma_3')
p = sp.Matrix([[p_00, p_01], [p_10, p_11]])
H = np.pi/2 *qp.sigmax()
pm = qp.sigmam()
pp = qp.sigmap()
pz = qp.sigmaz()

part_1 = (gamma_1)*(pm*p*pp - 1/2*(pp*pm*p + p*pp*pm))
part_2 = gamma_2*(pp*p*pm - 1/2*(pm*pp*p + p*pm*pp))
part_3 = gamma_3*(pz*p*pz - 1/2*(pz*pz*p + p*pz*pz))

res = part_1 + part_2 + part_3
res_n = sp.simplify(res)
#%%
t, gamma = sp.symbols('t, gamma', real = True)
hamiltonian = two_qubit_gate_rydberg_w_dark(d, o, 1, 1, 0.01)

val_o, vec = np.linalg.eig(-1J*hamiltonian)
val = np.exp(val_o)


val_o_n = val_o

for i in range(len(val_o)):
    if (np.real(val_o[i]) < 1e-10 and np.imag(val_o[i]) < 1e-10):
        val_o_n[i] = np.real(0.0) + np.imag(0.0)
    elif np.real(val_o[i]) < 1e-10:
        val_o_n[i] = np.real(0.0) + np.imag(val_o[i])
    elif np.imag(val_o[i]) < 1e-10:
        val_o_n[i] = np.imag(0.0) + np.real(val_o[i])

tmp = sp.Matrix(-1J*val_o*t)
tmp = tmp.applyfunc(sp.exp)
M = (sp.diag(*tmp))

v = sp.Matrix(vec)
v_m1 = sp.Matrix(np.linalg.inv(vec))
K_1 = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))
L_1 = gamma*sp.Matrix(np.kron(K_1, qp.identity(4)).reshape(16,16))

res = M @ dgr(v) @ L_1 @ v @ dgr(M)
res = sp.simplify(res)
#afterwards do dgr(v_M1)*EXPR*v_m1


#%%
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
plt.rcParams["figure.figsize"] = (18,6)

# Define the basis states (e.g., |0> and |1>)
basis_states = [basis(2, 0), basis(2, 1)]

ham = single_gate_hamiltonian_ryderg_fourlevels(0,1)

# Define the Pauli matrices (sigma_x, sigma_y, sigma_z)
sigmax = sigmax()
sigmay = sigmay()
sigmaz = sigmaz()

# Define the parameters
omega = 1.0 * 2 * np.pi  # Energy separation between levels
epsilon = 1 * omega    # Amplitude of the driving field

T = np.linspace(0.01,100* np.pi / epsilon, 230)

for i in range(0,1):
    # Time range for simulation
    timesteps = 1500
    t = np.linspace(0.0, T[-1], timesteps) #-1 to i if you want the whole video
    
    # Hamiltonian of the system
    H0 = 0.5 * omega * sigmaz
    H1 = epsilon * (sigmax + sigmay) / 2.0
    H = [H0, [H1, 'cos(omega*t)']]
    
    # Initial state: qubit starts in the ground state |0>
    psi0 = basis_states[0]
    psi0 = qp.fock([2, 2], [0, 0])
    
    linblads = 10e-1*sigmam()
    
    gamma_1 = 8e-2
    
    K_0_single = qp.Qobj(np.sqrt(gamma_1)*np.array(([0, 0, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0])), dims=[[2,2],[2,2]])

    # Time evolution
    result = mesolve(ham, psi0, t, [K_0_single], [qp.tensor(qp.Qobj([[1,0],[0,0]]), qp.Qobj([[1,0],[0,0]]))])
    result2 = mesolve(ham, psi0, t, [], [qp.tensor(qp.Qobj([[1,0],[0,0]]), qp.Qobj([[1,0],[0,0]]))])
    
    
    # Plot the Rabi oscillations
    fig, ax = plt.subplots()
    plt.plot(t, result2.expect[0], label=r'2$\langle\sigma_x\rangle$')

    plt.plot(t, result.expect[0], label=r'$\langle\sigma_x\rangle$', color = 'tab:red')
    
    plt.yticks([0,1])
    plt.xticks([])
    
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1] = r'$ 1\rangle$'
    labels[0] = r'$ 0\rangle$'
    
    T_o = 100* np.pi / epsilon  # Period of the driving field
    
    plt.xlabel('Time')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.xlim([0,T_o])
    ax.set_yticklabels(labels)
    plt.show()
    plt.savefig(f'/home/david/rabiboth.pdf', dpi=1000)
