# import the relevant libraries
import numpy as np
from scipy.linalg import svd
from ncon import ncon
from observables import entropy_per_site


class ctmrg:
    """ Class implementing the CTMRG algorithm. """

    def __init__(self,
        on_site_tensor,
        init_env_tensors,
        density_matrix,
        bond_dim,
        max_iter,
        tol,
        ):
        """
        Args:
            on_site_tensor (np.array): contain all the physical information of the system
            init_env_tensors (List[np.array]): initialization of the 8 environment tensors in the 
                                               following order: [C1,C2,C3,C4,T1,T2,T3,T4]
            density_matrix (str): either 'Orus' or 'Nishino'
            bond_dim (int): bond/truncation dimension of the indices
            max_iter (int): maximum number of CTMRG iterations
            tol (float): observable tolerance to stop the algorithm
        """
        self.on_site_tensor = on_site_tensor
        self.init_env_tensors = init_env_tensors
        self.density_matrix = density_matrix
        self.bond_dim = bond_dim
        self.max_iter = max_iter
        self.tol = tol
    
    def one_iar_move(self, on_site_tensor, env_tensors):
        """ Perform either a left, right, top, bottom insertion-absorption-renormalization move.
        In this function, we use the notation for the left move without loss of generality.

        Args:
            on_site_tensor (np.array): contain all the physical information of the system
            env_tensors (List[np.array]): current 8 environment tensors
        
        Return:
            env_tensors (List[np.array]): 8 environment tensors after the move
        """

        # Unpack the environment tensors and on-site tensor
        C1,C2,C3,C4,T1,T2,T3,T4 = env_tensors
        a = on_site_tensor

        # Insertion and absorption
        C1_tilde = ncon([C1,T1],[[-1,1],[1,-2,-3]])
        T4_tilde = ncon([T4,a],[[-1,1,-5],[-4,-3,-2,1]])
        C4_tilde = ncon([C4,T3],[[1,-3],[-1,-2,1]])
                
        # Convert the tensors of rank 3 to matrices for SVD
        d1, d2, d3 = C1_tilde.shape  
        mat_C1_tilde = C1_tilde.reshape((d1*d2,d3))
        mat_C4_tilde = C4_tilde.reshape((d1*d2,d3))
        
        if self.density_matrix == 'Nishino':
            # Isometry suggested by Nishino (1996):
            U = svd(ncon([mat_C1_tilde,C2,C3,mat_C4_tilde],[[-1,1],[1,2],[2,3],[-2,3]]))[0]
        elif self.density_matrix == 'Orus':
            # Isometry suggested by Orus (2009):
            U = svd(mat_C1_tilde @ mat_C1_tilde.conj().T + mat_C4_tilde @ mat_C4_tilde.conj().T)[0]
        else:
            print('This density matrix has not yet been implemented.')
            
        # Keep only the first chi singular vectors and reshape the resulting projector
        if self.bond_dim <= max(U.shape):
            U = U[:,:self.bond_dim].reshape((d1,d2,self.bond_dim))
        else:
            U = U.reshape((d1,d2,d1*d2))
            
        # Renormalization
        C1_prime = ncon([C1_tilde,U.conj()],[[1,2,-2],[1,2,-1]])
        T4_prime = ncon([U,T4_tilde,U.conj()],[[1,2,-3],[3,4,-2,2,1],[3,4,-1]])
        C4_prime = ncon([C4_tilde,U],[[-1,1,2],[2,1,-2]])
            
        return [C1_prime,C2,C3,C4_prime,T1,T2,T3,T4_prime]

    def one_ctmrg_iter(self, env_tensors):
        """ One CTMRG iteration consists of 4 insertion-absorption-renormalization moves.

        Args:
            env_tensors (List[np.array]): current 8 environment tensors

        Returns:
            env_tensors (List[np.array]): 8 environment tensors after one CTMRG iteration
        """
        a = self.on_site_tensor

        # Left, right, top, bottom moves respectively:
        env_tensors = self.one_iar_move(a, env_tensors) 
        env_tensors = self.one_iar_move(a.transpose(2,3,0,1), [env_tensors[i] for i in [2,3,0,1]+list(np.array([2,3,0,1])+4)])      
        env_tensors = self.one_iar_move(a.transpose(3,0,1,2), [env_tensors[i] for i in [3,0,1,2]+list(np.array([3,0,1,2])+4)]) 
        env_tensors = self.one_iar_move(a.transpose(2,3,0,1), [env_tensors[i] for i in [2,3,0,1]+list(np.array([2,3,0,1])+4)]) 

        # Normalize the environment tensors
        env_tensors = [env_tensor/np.amax(abs(env_tensor)) for env_tensor in env_tensors]

        return env_tensors

    def run(self):
        """ Run the CTMRG algorithm.
        
        Returns:
            env_tensors (List[np.array]): final 8 environment tensors
        """
        # Initialize relevant variables
        iter = 0
        entropies = [0,10] 
        env_tensors = self.init_env_tensors

        # While the stopping criteria are not met, continue the algorithm
        while iter < self.max_iter and np.abs(entropies[-1]-entropies[-2]) > self.tol:

            # Perform one CTMRG iteration
            env_tensors = self.one_ctmrg_iter(env_tensors)

            # After this CTMRG iteration, calculate observable(s)
            ent_per_site = entropy_per_site(self.on_site_tensor, env_tensors)
            entropies.append(ent_per_site)  

            # Update the iteration count and print the observable result(s)
            iter += 1
            print(f'Iteration: {iter}  Entropy per site: {ent_per_site}') if iter % 1 == 0 else None

        return env_tensors, entropies

