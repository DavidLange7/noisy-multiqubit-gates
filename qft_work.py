# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:25:46 2024

@author: david
"""
import numpy as np
import matplotlib.pyplot as plt
import qutip as qp
import sympy as sp
from sympy.physics.quantum.dagger import Dagger as dgr
from qutip import tensor, basis
import scipy
import pylab as plb
from qiskit import QuantumCircuit
from sympy.physics.quantum import TensorProduct
from functools import reduce
from itertools import product
import itertools

def single_gate_hamiltonian_ryderg_fourlevels(Delta, Omega, x_1 = 1, x_2 = 1):
    hamiltonian = np.array([[0, np.conj(Omega)/2*x_1, 0, 0], [Omega/2*x_2, -Delta, 0, 0], [0,0,1,0], [0,0,0,1]])
    hamiltonian = qp.Qobj(hamiltonian)
    
    return qp.Qobj(hamiltonian.full().reshape((4,4)),
            dims=[[2,2],[2,2]])

def rydberg_blockade_hamiltian(delta, omega, x_1, x_2, V):
    '''
    resulting order: 00, 01, 0r, 0d, 10, 11, 1r, 1d, r0, r1, rr, rd, d0, d1, dr, dd
    '''
    sigp = tensor(qp.Qobj([[0,0,1,0]]).dag(), qp.Qobj([[0,1,0,0]]))
    sigm = tensor(qp.Qobj([[0,1,0,0]]).dag(), qp.Qobj([[0,0,1,0]]))
    n = tensor(qp.Qobj([[0,0,1,0]]).dag(), qp.Qobj([[0,0,1,0]]))



    H_1 = omega/2*(sigm*x_2 + sigp*x_1) - delta*n
    A = qp.Qobj(qp.tensor(n, qp.identity(4))).full()
    B = qp.Qobj(qp.tensor(qp.identity(4), n)).full()

    H_2 = qp.tensor(H_1, qp.identity(4)).full() + qp.tensor(qp.identity(4), H_1).full() + V*(A@B)
    #H_2 = qp.Qobj(H_2)
    
    return H_2


def qft(circuit, n):
    """Apply QFT on the first n qubits in circuit."""
    for i in range(n):

        circuit.h(i)

        for j in range(i + 1, n):
            circuit.cp(np.pi / (2 ** (j - i)), j, i)

    for i in range(n // 2):
        circuit.swap(i, n - i - 1)
    return circuit

'''
circuit implementation on an N-qubit system
'''
def lindblad_terms(operators, num_qubits):
    
    identity = np.identity(4)
    linbs = []

    for operator in operators:
        
        linb_terms = [
            qp.Qobj(
                reduce(np.kron, [operator if i == target else identity for i in range(num_qubits)]),
                dims=[[4]*num_qubits, [4]*num_qubits]
            )
            for target in range(2)
        ]
        linbs.append(linb_terms)
    
    return linbs, len(linbs)

def swap_gate_matrix(num_qubits, qubit_1, qubit_2):
    
    swap_matrix = np.eye(2**num_qubits)
    
    basis_states = list(product([0, 1], repeat=num_qubits))
    
    for i, state in enumerate(basis_states):
        swapped_state = list(state)
        swapped_state[qubit_1], swapped_state[qubit_2] = swapped_state[qubit_2], swapped_state[qubit_1]
        
        swapped_index = int("".join(map(str, swapped_state)), 2)
        
        swap_matrix[i, i] = 0
        swap_matrix[i, swapped_index] = 1
    
    return swap_matrix


def singleq_gate(num_qubits, target, params = []):
    
    delta, omega, x_1, x_2 = params
    H_oneq = single_gate_hamiltonian_ryderg_fourlevels(delta, omega, x_1, x_2)
    identity = np.identity(4)
    
    H_full = qp.Qobj(
            reduce(np.kron,
    [H_oneq.full() if i == target else identity
        for i in range(num_qubits)]
    ),
    dims=[[4] * num_qubits, [4] * num_qubits]
    )
    
    return H_full

def twoq_ham(num_qubits, control, target, params = []):
    '''
    Notes: min(target, control) and max(target, control) used to not double count 
    '''
    delta, omega, x_1, x_2, V = params
    H_twoq = rydberg_blockade_hamiltian(delta, omega, x_1, x_2, V)
    identity = np.identity(4)
    
    if (abs(control - target) != 1):
        raise ValueError('control and target need to be adjacent qubits')
        
    if num_qubits == 2:
        H_full = qp.Qobj(H_twoq, dims=[[4]*num_qubits, [4]*num_qubits])
        
    else:
        H_full = qp.Qobj(
                reduce(np.kron,
        [H_twoq if i == min(target, control) else identity
            for i in range(num_qubits)
            if i != max(target, control)]
    ),
    dims=[[4] * num_qubits, [4] * num_qubits]
    )
    
    return H_full
    
def generate_basis(num_qudits, levels, pattern):
    '''
    such as generate_basis(3, ["0","1","r","d"] , '11x')
    '''

    basis_states = ["".join(comb) for comb in itertools.product(levels, repeat=num_qudits)]
    state_dict = {index: state for index, state in enumerate(basis_states)}
    output_strings = [pattern.replace('x', str(x)) for x in levels]    
    
    matching_indices = [[index for index, state in state_dict.items() if output_strings[i] in state] for i in range(len(levels))]
    
    return state_dict

def expectation_value(dens_mats, indx_sub, num_qudits):
    '''
    e.g. if you want <x11> then you would use indx_sub = 5
         if <xdd> then -1 and so on
    '''
    
    expectation = np.zeros([len(dens_mats)])
    
    projector = np.zeros([16,16])
    projector[indx_sub, indx_sub] = 1
    for i in range(len(dens_mats)):
        rho_i = dens_mats[i]

        if num_qudits == 2:

            expectation_i = np.trace(rho_i.full() @ projector).real
            expectation[i] = expectation_i

        else:
        
            rho_i = qp.Qobj(rho_i, dims=[[4]*num_qudits, [4]*num_qudits])
            reduced_rho = rho_i.ptrace([i for i in range(0, 2)])
            expectation_i = np.trace(reduced_rho.full() @ projector).real
            expectation[i] = expectation_i
    
    return expectation
#%%
# Time discretization
initial_time = 0.0
final_time = 300 # You get a rabi oscillation for each dt=1.
num_timesteps = int(final_time)*50
times = np.linspace(initial_time, final_time, num_timesteps)

num_qubits = 3

psi0 = qp.fock([4]*num_qubits, [1]*num_qubits) 

dim = 4**num_qubits
sq_target = 1

o_p, t, d, o, gam1, gam2, gamr = sp.symbols('o_p, t, d, o, gam1, gam2, gamr', real = True)
x1, x2 = sp.symbols('x1, x2')

t1 = 0.2 #amplitude damping
t2 = 300*10**(-3) #p.deph damping

o_real = 2*np.pi*10*10**(3)
o_fin = np.pi

tg = np.pi/o_real

gamma_1 = tg/t1
gamma_2 = tg/t2

t1 = 50*10**(-6) #amplitude damping

tg = 0.5*10**(-6)

gamma_1 = tg/t1


K_1 = np.array(([0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]))



d = 0
o = o_fin
o_p = np.sqrt(d**2 + o**2)

x_1 = 1
x_2 = 1

#%%
lindblads = lindblad_terms([K_1], num_qubits)
K = np.reshape(lindblads[0], (2)*lindblads[1]).tolist()

params = [0, np.pi, 1, 1, 5]

H = twoq_ham(num_qubits, 0, 1, params = params)
#H = singleq_gate(num_qubits, 1, params = params[:-1])

U = scipy.linalg.expm(-1J*H.full())


linb = [qp.Qobj(np.sqrt(gamma_1)*(np.conjugate(U).T @ (K[j].full() @ U)), dims=[[4]*num_qubits, [4]*num_qubits]) for j in range(len(K))]


#i, i2, i3, i4, i5, i6 = 5, 21, 37, 53, -1, 0
#i, i2, i3, i4, i5, i6 = 0, 1, 2, 3, 4, 5

result = qp.mesolve(
                    H     = H,
                    rho0  = psi0,
                    tlist = times, 
                    c_ops = linb,
                    options =  {"store_states": True}
                    )

full_states = result.states

full_evolution = False

fig, ax = plt.subplots()
if full_evolution:
    mask = np.full(num_timesteps, True)
else:
    frequency = num_timesteps/(final_time-initial_time)
    mask = (np.arange(num_timesteps) % frequency) == 0
    
plt.plot(expectation_value(full_states, 5, num_qubits)[mask])
    

#%% Example using qiskit
n_qubits = 5
qc = QuantumCircuit(n_qubits)
qft(qc, n_qubits)
qc.draw('mpl')
#%%
#HERE we generate the Hadamard gate yuppi

def single_gate(num_qubits, target, params):
    
    delta, omega, x_1, x_2 = params
    hamiltonian = singleq_gate(num_qubits, target, params)
    return (-1j * hamiltonian).expm()  # Time evolution for t = 1


x_gate = single_gate(1, 0, params = [0, np.pi, 1, 1])
ysqrt_gate = single_gate(1, 0, params = [0, np.pi/2, -1J, 1J])

hadamard = x_gate @ ysqrt_gate
print(1J*hadamard[:2, :2])
ideal_hadamard = qp.Qobj(
    (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
    dims=[[2], [2]]
)
#%%
#HERE we generate the CNOT gate yuppix2
import qutip as qp

def Cz_gate(num_qubits, control, target, psi, params):
    '''
    resulting order: 00, 01, 0r, 0d, 10, 11, 1r, 1d, r0, r1, rr, rd, d0, d1, dr, dd
    '''
    delta, omega, x_1, x_2, V, eps, tau = params
    times = np.linspace(0.0, tau, 200)
    
    hamiltonian = twoq_ham(num_qubits, control, target, params[:-2])
    U = scipy.linalg.expm(-1J*hamiltonian.full())
    psi1 = U @ psi.full()
    
    result = qp.mesolve(hamiltonian, psi, times, [], [])
    psi = result.states[-1]
    
    params[1] = omega*np.exp(1J*eps)
    hamiltonian = twoq_ham(num_qubits, control, target, params[:-2])
    U = scipy.linalg.expm(-1J*hamiltonian.full())
    psi1 = U @ psi1
    
    result = qp.mesolve(hamiltonian, psi, times,[], [])
    psi = result.states[-1] 
    
    return psi, psi1

psi = np.zeros(16)
psi[5] = 1 #|11>
psi0 = qp.fock([4, 4], [1, 1])

Omega   = 1
frac_DO = 0.377371
prod_Ot = 4.29268
Delta = frac_DO * Omega 
tau = prod_Ot / Omega


res = Cz_gate(2, 0, 1, psi0, params = [Delta, Omega, 1, 1, 50, 3.90242, tau])

rho = (np.outer(np.conj(res[0].full()), res[0].full()))


import seaborn as sns
plt.figure(1)
sns.heatmap(np.real(rho), annot = True, fmt=".2f", cmap="YlGnBu")
plt.title("cnot_operator matrix")
plt.show()

rho = (np.outer(np.conj(res[1]), res[1]))
plt.figure(2)
sns.heatmap(np.real(rho), annot = True, fmt=".2f", cmap="YlGnBu")

