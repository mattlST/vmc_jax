import jax

import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial
from jVMC.global_defs import tCpx
from jVMC.operator import BranchFreeOperator, scal_opstr
opDtype = tCpx


#### operators ####
def Id(idx=0, lDim=2):
    """Returns an identity operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining an identity operator

    """

    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)], dtype=np.int32),
            'matEls': jnp.array([1. for j in range(lDim)], dtype=opDtype), 'diag': True}

def create(idx=0, lDim=2):
    """Returns an creation operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining an identity operator

    """

    return {'idx': idx, 'map': jnp.array([j+1 for j in range(lDim-1)]+[0], dtype=np.int32),
            'matEls': jnp.array([np.sqrt(j+1) for j in range(lDim-1)]+[0.], dtype=opDtype), 'diag': False}

def destroy(idx=0, lDim=2):
    """Returns an annihlation operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining an annihlation operator

    """

    return {'idx': idx, 'map': jnp.array([0]+[j-1 for j in range(1,lDim)], dtype=np.int32),
            'matEls': jnp.array([0.]+ [np.sqrt(j) for j in range(1,lDim)], dtype=opDtype), 'diag': False}
def number(idx=0, lDim=2):
    """Returns a number operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining a number operator

    """

    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)], dtype=np.int32),
            'matEls': jnp.array([j for j in range(lDim)], dtype=opDtype), 'diag': True}



def interaction(idx=0, lDim=2):
    """Returns a number operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining a number operator

    """

    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)], dtype=np.int32),
            'matEls': jnp.array([j*(j-1) for j in range(lDim)], dtype=opDtype), 'diag': True}

################

def BoseHubbard_Hamiltonian1D(L,J,U,lDim=2,mu=0,V=0):
    """
    L: number of sites
    J: next-neighbour hopping
    U: interaction
    mu: chemical potentials
    V: non-local next-neighbour interaction
    """
    if not hasattr(mu, "__len__"):
        mu = [mu]*L
    if not hasattr(V, "__len__"):
        V = [V]*L    
    hamiltonian1D = BranchFreeOperator(lDim=lDim)
    for l in range(L):    
        hamiltonian1D.add(scal_opstr(-J, (create(l,lDim), destroy((l + 1) % L,lDim))))
        hamiltonian1D.add(scal_opstr(-J, (create((l+1)%L,lDim), destroy((l),lDim))))
        
        hamiltonian1D.add(scal_opstr(U/2., (number(l,lDim ),number(l,lDim )) ))
        hamiltonian1D.add(scal_opstr(-U/2., (number(l,lDim ),) ))
        if np.linalg.norm(mu)>1e-10:
            hamiltonian1D.add(scal_opstr(mu[l], (number(l,lDim ),) ))
        if np.linalg.norm(V)>1e-10:
            hamiltonian1D.add(scal_opstr(V[l], (number(l,lDim), number((l + 1) % L,lDim))))
    return hamiltonian1D

def interactionTerm(L,lDim=2):
    iTerm = BranchFreeOperator(lDim=lDim)
    for l in range(L):    
        iTerm.add(scal_opstr(1., (number(l,lDim ),number(l,lDim )) ))
        iTerm.add(scal_opstr(1., (number(l,lDim ),) ))
    return iTerm
    
def hoppingTerm(L,lDim=2):
    jTerm = BranchFreeOperator(lDim=lDim)
    for l in range(L):    
        hamiltonian1D.add(scal_opstr(-1, (create(l,lDim), destroy((l + 1) % L,lDim))))
        hamiltonian1D.add(scal_opstr(-1, (create((l+1)%L,lDim), destroy((l),lDim))))
        
    return jTerm
def occupations(L,lDim=2):
    occ= [BranchFreeOperator(lDim=lDim) for l in range(L)]
    for l in range(L):    
        occ[l].add(scal_opstr(1, (n(l,lDim), )))     
    return occ


################


def propose_hopping(key, s, info,particles):
    # propose hopping of a single particle from one site to a random site 
    idxKeyDestroy, idxKeyCreate = jax.random.split(key, num=2)
    # can't use jnp.where because then it is not jit-compilable
    # find indices based on cumsum
    bound_destroy = jax.random.randint(idxKeyDestroy, (1,), 1,  particles + 1)
    #bound_down = jax.random.randint(idxKeyDown, (1,), 1,  particles  + 1)

    id_destroy = jnp.searchsorted(jnp.cumsum(s), bound_destroy)
    #id_down = jnp.searchsorted(jnp.cumsum(particles - s), bound_down)

    idx_destroy = jnp.unravel_index(id_destroy, s.shape)
    
    #Does it matter if I end up at the same site?
    #if not 
    #id_create = jax.random.randint(idxKeyCreate, (1,), 0,  s.size) # can create anywhere a particle, does not matter for bosons.

    #if yes 
    id_create = jax.random.randint(idxKeyCreate, (1,), 0,  s.size-1)
    id_create = id_create+(id_create>=id_destroy)
    
    
    idx_create = jnp.unravel_index(id_create, s.shape)
    

    s = s.at[idx_destroy].add(-1)
    s = s.at[idx_create].add(1)
    #print(idx_up)
    #print(idx_down)
    
    return s

def propose_hopping_nn(key, s, info,particles,L):
    # propose hopping of a single particle from one site to a random site 
    idxKeyDestroy, idxKeyCreate = jax.random.split(key, num=2)
    # can't use jnp.where because then it is not jit-compilable
    # find indices based on cumsum
    bound_destroy = jax.random.randint(idxKeyDestroy, (1,), 1,  particles + 1)
    #bound_down = jax.random.randint(idxKeyDown, (1,), 1,  particles  + 1)

    id_destroy = jnp.searchsorted(jnp.cumsum(s), bound_destroy)
    #id_down = jnp.searchsorted(jnp.cumsum(particles - s), bound_down)

    idx_destroy = jnp.unravel_index(id_destroy, s.shape)
    
    #Does it matter if I end up at the same site?
    #if not 
    #id_create = jax.random.randint(idxKeyCreate, (1,), 0,  s.size) # can create anywhere a particle, does not matter for bosons.

    #if yes 
    id_create = jax.random.randint(idxKeyCreate, (1,), -1,  2)
    id_create = (id_destroy+id_create)%L
    
    
    idx_create = jnp.unravel_index(id_create, s.shape)
    

    s = s.at[idx_destroy].add(-1)
    s = s.at[idx_create].add(1)
    #print(idx_up)
    #print(idx_down)
    
    return s
