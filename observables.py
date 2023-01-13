# import the relevant libraries
import numpy as np
from ncon import ncon


def entropy_per_site(on_site_tensor, env_tensors):
    """ Compute the entropy per site.

    Args:
        on_site_tensor (np.array): contain all the physical information of the system
        env_tensors (List[np.array]): 8 environment tensors in the 
                                      following order: [C1,C2,C3,C4,T1,T2,T3,T4]

    Returns:
        entropy per site
    """

    # Unpack the environment tensors
    C1,C2,C3,C4,T1,T2,T3,T4 = env_tensors

    # Get the partition function
    Z = ncon([C1,T1,C2,T4,on_site_tensor,T2,C4,T3,C3], [[1,2],[2,3,4],[4,5],[8,6,1],[3,7,9,6],[5,7,10],[11,8],[12,9,11],[10,12]])

    # Get the exponential of the entropy (kappa)
    kappa = Z*ncon([C1,C2,C4,C3],[[1,2],[2,3],[4,1],[3,4]])
    kappa /= ncon([C1,C2,T4,T2,C4,C3],[[1,2],[2,3],[5,4,1],[3,4,6],[7,5],[6,7]])
    kappa /= ncon([C1,T1,C2,C4,T3,C3],[[1,2],[2,3,4],[4,5],[6,1],[7,3,6],[5,7]])

    if kappa < 0:
        return 0
    else:
        return np.log(kappa)