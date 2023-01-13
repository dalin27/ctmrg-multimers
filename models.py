# import the relevant libraries
import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon


def multimer_on_site_tensor(k):
    """ Implement the on-site tensor for multimer systems.

    Args:
        k (int): size of the multimers

    Return:
        Q (np.array): intermediary on-site tensor
        P (np.array): tensor mediating the interaction between lattice sites
        a (np.array): on-site tensor
    """

    # (Intermediary) on-site tensor
    Q= np.zeros((k,k)) 
    Q[0,0] = 1 
    
    # Tensor mediating the interaction between lattice sites
    P = np.zeros((k,k,k,k))
    
    if k % 2 == 0: # k even
        
        Q[k-1,k-1] = 1
        for i in np.arange(2,k,2):
            Q[i-1,i] = Q[i,i-1] = 1
            
        for i in np.arange(1,k+1,2):
            P[i-1,0,i,0] = P[i,0,i-1,0] = 1 # horizontal multimers
            P[0,i-1,0,i] = P[0,i,0,i-1] = 1 # vertical multimers
            
    else: # k odd
        
        for i in np.arange(2,k+1,2):
            Q[i-1,i] = Q[i,i-1] = 1
        
        for i in np.arange(1,k,2):
            P[i-1,0,i,0] = P[i,0,i-1,0] = P[k-1,0,k-1,0] = 1 # horizontal multimers
            P[0,i-1,0,i] = P[0,i,0,i-1] = P[0,k-1,0,k-1] = 1 # vertical multimers

    a = ncon([P,sqrtm(Q),sqrtm(Q),sqrtm(Q),sqrtm(Q)], [[1,2,3,4],[1,-1],[2,-2],[3,-3],[4,-4]]) 
    
    return Q, P, a

