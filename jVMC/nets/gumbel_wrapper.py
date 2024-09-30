import jax
import jax.numpy as jnp
import jax.random as jrnd
from functools import partial
#import jVMC.global_defs as global_defs
from flax import linen as nn
from typing import Any, List, Optional, Tuple
from jax import Array, vmap, jit


@jit
def sorting_gumble(sample,logits,gumbel,states):
    indexes = jnp.argsort((-gumbel),axis=None)#.reshape(shape_gumbel)
    numSamples = sample.shape[0]
    LocalHilDim = sample.shape[1]
    L = sample.shape[2]
    #jax.debug.print("shape {x},{y},{z}",x=numSamples,y=LocalHilDim,z=L)
    indexes_states = (indexes // LocalHilDim)[:numSamples]
    sample = sample.reshape(-1,L)[indexes]
    sample = sample.reshape(LocalHilDim,numSamples,L)
    sample = jnp.swapaxes(sample,0,1)
    
    logits = logits.ravel()[indexes]
    logits = logits.reshape(LocalHilDim,numSamples).T
    
    gumbel = gumbel.ravel()[indexes]
    gumbel = gumbel.reshape(LocalHilDim,numSamples).T
    vals, treedef  = jax.tree_util.tree_flatten(states)
    #jax.debug.print("states shape: {x}",x=states[0].shape)
    #jax.debug.print("vals: {x}",x=vals)
    #jax.debug.print("treedef: {x}",x=treedef)
    vals_ordered = [v[indexes_states] for v in vals]
    states = jax.tree_util.tree_unflatten(treedef,vals_ordered)
    #jax.debug.print("new states shape: {x}",x=states[0].shape)
    
    return sample,logits,gumbel,states

class gumbel_wrapper(nn.Module):
    """
    Wrapper module for symmetrization.
    This is a wrapper module for the incorporation of lattice symmetries. 
    The given plain ansatz :math:`\\psi_\\theta` is symmetrized as

        :math:`\\Psi_\\theta(s)=\\frac{1}{|\\mathcal S|}\\sum_{\\tau\\in\\mathcal S}\\psi_\\theta(\\tau(s))`

    where :math:`\\mathcal S` denotes the set of symmetry operations (``orbit`` in our nomenclature).

    Initialization arguments:
        * ``orbit``: orbits which define the symmetry operations (instance of ``util.symmetries.LatticeSymmetry``)
        * ``net``: Flax module defining the plain ansatz.
        * ``avgFun``: Different choices for the details of averaging.

    """
    #orbit: LatticeSymmetry
    net: callable
    is_gumbel = True

    #avgFun: callable = avgFun_Coefficients_Exp
    def setup(self):
        self.L = self.net.L
        self.LocalHilDim = self.net.LocalHilDim
        if hasattr(self.net, 'is_particle'):
            self.is_particle = self.net.is_particle
        else:
            self.is_particle = False
    def __post_init__(self):

        super().__post_init__()
    
    def __call__(self,*args,**kwargs):
        
        return self.net(*args,**kwargs)
        
    def _apply_fun(self, *args,**kwargs):
        return self.net.apply(*args,**kwargs)
    
    def _gumbel_step(self,sample,logits,gumbel,key,states,position):        
        #new samples with (0,..,LocalHilDim-1) at position
        #sample = jnp.array([sample[0].at[position].set(l) for l in jnp.arange(self.LocalHilDim)])
        #right shifted input
        #inputt = jnp.array([jnp.pad(sample[0,:-1],(1,0))])
        logitnew = jnp.zeros_like(logits)
        sample = jnp.array([sample[0].at[position].set(l) for l in jnp.arange(self.LocalHilDim)])
        #right shifted input
        inputt = jnp.array([jnp.pad(sample[0,:-1],(1,0))])

        #jax.debug.print("position: {x}",x=position)
        #jax.debug.print("inputt: {x}",x=inputt)

        #jax.debug.print("inputt[pos]: {x}",x=inputt[:,position])
        if self.is_particle:
            cumsum = jnp.sum(inputt+jnp.abs(inputt))//2
            #if self.net.net.__name__=="GPT":
            #    # no shift for the GPT model
            #    #states = (jnp.concatenate((sample[0,:position],[-1]*L-position)),position)
            #    logitnew, next_states = self(jnp.expand_dims(sample[0,position-1],0),block_states = states, output_state=True,cumsum=cumsum,position=position)            
            #else:
            logitnew, next_states = self(inputt[:,position],block_states = states, output_state=True,cumsum=cumsum,position=position)
        else:
            logitnew, next_states = self(inputt[:,position],block_states = states, output_state=True)
        #jax.debug.print("logits new: {x}",x=logitnew)
        logitnew = logits[0] + logitnew 
        
        gumbelnew = logitnew + jrnd.gumbel(key[0],shape=(self.LocalHilDim,)) 
        
        Z = jnp.nanmax(gumbelnew)
        gumbelnew = jnp.nan_to_num(-jnp.log(
            jnp.exp(-gumbel[0])-jnp.exp(-Z)+jnp.exp(-gumbelnew) 
            ),nan=-jnp.inf)

        #gumbelnew = gumbelnew
        return sample, logitnew, gumbelnew, next_states
    
    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        
        """
        def generate_sample(key):
            key = jrnd.split(key, self.net.L)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True)

            choice = jrnd.categorical(key[0], logits.ravel()) # abide by the indexing convention and apply -1
            s_cumsum = self.net.Q - choice # create cumsum of the quantum number
            _, s = self._scanning_fn((jnp.expand_dims(choice,0),carry,s_cumsum),(key[1:],jnp.arange(1,self.net.L)))
            return jnp.concatenate([jnp.expand_dims(choice,0),s])
        """
        # get the samples
        keys = jrnd.split(key, (self.net.L))
        ## init stap
        shape_samples = (numSamples,self.LocalHilDim,self.net.L)
        shape_logits = (numSamples,self.LocalHilDim)
        shape_gumbel = (numSamples,self.LocalHilDim)
        #print(shape_samples,shape_logits)
        working_space_samples = jnp.full(shape_samples,-2,dtype=jnp.int64)
        working_space_logits = jnp.full(shape_logits,-jnp.inf,dtype=jnp.float64)
        working_space_gumbel = jnp.full(shape_gumbel,-jnp.inf,dtype=jnp.float64)
        
        #working_space_samples = working_space_samples.at[0,0,0].set(0)
        working_space_logits = working_space_logits.at[0,0].set(0.)
        working_space_gumbel = working_space_gumbel.at[0,0].set(0.)
        
        states = None
        init_work = partial(self._gumbel_step, position=0,states=states)
        key0 = jrnd.split(keys[0],numSamples)
        key0 = jnp.expand_dims(key0,-2)
        samples,logits,gumbel,states  = jax.vmap(init_work)(working_space_samples,working_space_logits,working_space_gumbel,key0)
        
        init_carry = sorting_gumble(samples,logits,gumbel,states)
        
        res,_ = self._scanning_fn(init_carry,(keys[1:],jnp.arange(1,self.net.L)))
        samples, logits,gumbels,_ = res
        
        kappa = gumbels[0,1]

        re_weights = jnp.nan_to_num(jnp.exp(logits[:,0]) /(-jnp.expm1(-jnp.exp(logits[:,0]-kappa))),0)
        
        return samples[:,0,:],logits[:,0]*self.net.logProbFactor,re_weights/jnp.sum(re_weights),kappa
        #return samples[:,0,:],logits[:,0],re_weights/jnp.sum(re_weights)

    @partial(nn.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry, key):
        position = key[1]
        sample = carry[0]
        #jax.debug.print("{x}", x=sample)

        logits = carry[1]
        gumbel = carry[2]
        states = carry[3]
        keys = jrnd.split(key[0],carry[0].shape[0])
        keys = jnp.expand_dims(keys,-2)

        p_workN = partial(self._gumbel_step,position=position)
        sample,logits,gumbel,states = jax.vmap(p_workN)(sample,logits,gumbel,keys,states)
        #jax.debug.print("gumbelnew new {x}", x=gumbel)

        #### sorting gumble value
        return sorting_gumble(sample,logits,gumbel,states),None
        