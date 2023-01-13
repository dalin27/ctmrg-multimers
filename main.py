# import the relevant libraries
from models import multimer_on_site_tensor
from ctmrg_algorithm import ctmrg
from ncon import ncon
from utils import kron_delta
import numpy as np

# Set global variables
k = 7
bond_dim = 600
max_iter = 30

# Get the on-site tensor of k-mers
_,_, a = multimer_on_site_tensor(k)

# # Randomly initialize the environment tensors
# np.random.seed(0)
# C1 = C2 = C3 = C4 = np.random.randn(k,k) 
# T1 = T3 = T2 = T4 = np.random.randn(k,k,k)
# init_env_tensors = [C1,C2,C3,C4,T1,T2,T3,T4]

# # Initialize the environment tensors using the on-site tensor 
# C = ncon([kron_delta(2, k), a], [[1,2], [1,2,-1,-2]])
# T = ncon([kron_delta(1, k), a], [[1], [1,-1,-2,-3]])
# init_env_tensors = [C]*4 + [T]*4

# Initialize the environment tensors using tensors obtained from previous simulations
init_env_tensors = np.load('results/env_tensors_k=7_chi=500_maxiter=50.npz')
C1,C2,C3,C4,T1,T2,T3,T4 = [init_env_tensors[x] for x in ['C1','C2','C3','C4','T1','T2','T3','T4']]
init_env_tensors = [C1,C2,C3,C4,T1,T2,T3,T4]

# Initialize the CTMRG algorithm
ctmrg_algo = ctmrg(        
    on_site_tensor=a,
    init_env_tensors=init_env_tensors,
    density_matrix='Nishino',
    bond_dim=bond_dim,
    max_iter=max_iter,
    tol=-1,
)

# Run the CTMRG algorithm
env_tensors, entropies = ctmrg_algo.run()
C1,C2,C3,C4,T1,T2,T3,T4 = env_tensors

np.savez('results/env_tensors_k={}_chi={}_maxiter={}.npz'.format(k,bond_dim,max_iter),
         C1=C1,C2=C2,C3=C3,C4=C4,T1=T1,T2=T2,T3=T3,T4=T4, entropies=entropies)
