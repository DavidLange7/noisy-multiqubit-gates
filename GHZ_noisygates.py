# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:37:11 2024

@author: idavi
"""

# This code is part of qmatchatea.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Using Qcircuit and qmatchatea to simulate qudits
================================================

This is an example for the advanced usage of the library.

Here we are going to construct our circuits and the operators to run
simulation on 3-level systems. This is completely general, and could
be used for n-level systems.

"""

# Import necessary modules
import numpy as np

from qmatchatea import run_simulation
from qmatchatea.circuit import Qcircuit, QCOperation, QCSwap
from qtealeaves.observables import TNObservables, TNObsProjective

import sys
sys.path.append("/Users/idavi/Desktop/qc_paper/noisy-multiqubit-gates")
import noisygate_rydberg_numerical as ng

import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce


#%%    
def plotting(mat):
    
    basis_labels = ['00', '01', '0r', '0d', 
                   '10', '11', '1r', '1d', 
                   'r0', 'r1', 'rr', 'rd', 
                   'd0', 'd1', 'dr', 'dd']
    
    plt.figure('real')
    sns.heatmap(np.real(mat), annot = True, fmt=".2f", cmap="YlGnBu",
                yticklabels=basis_labels, 
                xticklabels=basis_labels, 
    annot_kws={"size": 15},  # Font size for annotations
    cbar_kws={"shrink": 0.8} # Adjust colorbar size (optional)
                
    )
    plt.yticks(fontsize=14, rotation=0)
    plt.xticks(fontsize=14, rotation=0)
    plt.title("cz_operator matrix", fontsize = 14)
    
def multiq_gate(gate, control, target, num_qudits):
    
    tmp = len(gate)

    identity = np.identity(4)
    
    if tmp == 4:
        
        multi_gate = reduce(np.kron,
        [gate if i == target else identity
            for i in range(num_qudits)]
        )
    
    elif tmp == 16:
        
        if num_qudits == 2:
            multi_gate = gate
            
        else:
            multi_gate = reduce(np.kron,
            [gate if i == min(target, control) else identity
                for i in range(num_qudits)
                if i != max(target, control)]
        )
    
    else:
        raise ValueError('gate input is invalid, needs to be either 4x4 or 16x16')
        
    return multi_gate

local_dim = 4
num_sites = 3

###########################################
'''
#HADAMARD NOISY GATE
These are the parameters that implement a hadamard gate using the rydberg hamiltonian.
'''
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
h_noisy = 1J*x_gate @ ysqrt_gate

#################################################

def Cnot_gate(plot = False):

    K_1_single = np.array(([0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0]))
    x1 = 1
    x2 = 1
    
    eps = 1.894
    omega = 5*2*np.pi
    V = 1000
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
    
    h_multi = multiq_gate(h_noisy, 0, 1, 2)

    cnot = h_multi @ cz @ h_multi
    
    if plot == True:
        plotting(cz)
    
    return cnot


################################################################################
# We now define the circuit, which is an anagolous of the GHZ circuit, but
# defined on qudits where the two states in a superposition are the
# state 0 and 1
#David: Here is the main part, we create the algorithm here, by inputting the wanted single 
#and two body matrices thereby generating the circuit


qc = Qcircuit(num_sites=num_sites, local_dim=local_dim)

h_op = QCOperation("h", lambda: h_noisy)
qc.add(h_op, 0)
    
print("noisy gate H gate:", h_noisy[:2,:2])
print("#########################################")


cnot = Cnot_gate()
cnot_op = QCOperation("not", lambda: cnot)


for ii in range(num_sites - 1):
    qc.add(cnot_op, [ii + 1, ii])

################################################################################
# We can then add the specific observable to the class :py:class:`TNObservables`,
# which is the one devoted to the input/output/measurement management.
# Notice that all the observables are actually defined in qtealeaves.

observables = TNObservables()

################################################################################
# Here we define the projective measurements. The only parameter is the number
# of shots, that in the experiment would be the number of repetitions.
# Adding it to the observables is easy, using the += pythonic syntax

num_shots = 1024
proj_meas = TNObsProjective(num_shots=num_shots)
observables += proj_meas

################################################################################
# Now we run the simulation passing all the parameters seen in this example.
# There are many more parameters available that are set as default here!

results = run_simulation(
    qc,
    local_dim=local_dim,
    observables=observables,
)

################################################################################
# To retrieve the projective measurement observables, we access results.measures

observable_measures = results.measures

print("-" * 30, "Observables results", "-" * 30)
for state, num_times_measured in observable_measures.items():
    print(f"State {state} measured {num_times_measured} times")

str_val = "".join([str(1)] * num_sites)
print(
    f"Expected 0000000000 around {num_shots//2} times and {str_val} around {num_shots//2} times"
)
print()

################################################################################
# There are some other runtime statistics saved by qmatchatea:
#
# - results.computational_time. The time spent in the circuit simulation in
#   seconds. Available for both python and fortran;
# - results.observables["measurement_time"]. Time spent in the measurement
#   process in seconds. Available only in python;
# - results.observables["memory"]. Memory used during the circuit simulation
#   in Gigabytes. It is a list of values. Here we just look at the memory peak,
#   i.e. its maximum. Available only in python;
# - results.fidelity. Lower bound on the fidelity of the state
# - results.date_time. yy-mm-dd-hh:mm:ss of the start of the simulation

comp_time = np.round(results.computational_time, 3)
meas_time = np.round(results.observables.get("measurement_time", None), 3)
memory = np.round(np.max(results.observables.get("memory", [0])), 4)
print("-" * 30, "Runtime statistics", "-" * 30)
print(f"Datetime of the simulation: {results.date_time}")
print(f"Computational time: {comp_time} s")
print(f"Measurement time: {meas_time} s")
print(f"Maximum memory used: {memory} GB")
print(
    f"Lower bound on the fidelity F of the state: {results.fidelity}, i.e.  {results.fidelity}≤F≤1"
)