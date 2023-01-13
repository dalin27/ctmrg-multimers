# import the relevant libraries
import numpy as np

def kron_delta(n, idx_dim):
    """ Define an n-dimensional kronecker delta function
    
    Args:
        n (int): rank of the kronecker delta  
        idx_dim (int): dimension of each index
    
    Returns:
        kdelta (np.array): kronecker delta tensor
    """
    kdelta = np.zeros(np.full(n, idx_dim))
    for i in np.arange(0, idx_dim, 1):
        kdelta[list(i for _ in range(n))] = 1
    return kdelta