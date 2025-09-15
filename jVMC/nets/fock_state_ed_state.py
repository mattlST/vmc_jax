##############################################
# J Rigo
# rigojonas@gmail.com
# Regensburg 20/11/2023
##############################################

import jax
import flax
#from flax import nn
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.nets.initializers import init_fn_args

from functools import partial

import jVMC.nets.initializers


class Target_fock(nn.Module):
    """Target wave function, returns a vector with the same dimension as the Hilbert space

        Initialization arguments:
            * ``L``: System size
            * ``d``: local Hilbert space dimension
            * ``delta``: small number to avoid log(0)

    """
    L: int
    fockStateIndex: int 
    lDim: int = 2
    delta: float = 1e-15
    alpha: float = 1e1
    def setup(self):
        self.fockState2 = (self.fockStateIndex // self.lDim**jnp.arange(self.L,dtype=int))%self.lDim
    @nn.compact
    def __call__(self, s):
        kernel = self.param('kernel',
                            nn.initializers.constant(1,dtype=global_defs.tReal),
                            (int(2**self.L)))

        #overlap_cl = s.T@self.fockState2
        overlap_qm = jnp.exp(-(self.alpha*(s-self.fockState2)**2).sum())
        return jnp.log(overlap_qm +self.delta)
    
        kernel = self.param('kernel',
                            nn.initializers.constant(1,dtype=global_defs.tReal),
                            (int(self.d**self.L)))
        # return amplitude for state s
        idx = ((self.d**jnp.arange(self.L)).dot(s)).astype(int)
        return jnp.log(abs(kernel[idx]+self.delta)) + 1.j*jnp.angle(kernel[idx]) 

    def get_fockstate(self):
        fockState2 = (self.fockStateIndex // self.lDim**jnp.arange(self.L,dtype=int))%self.lDim

        return fockState2