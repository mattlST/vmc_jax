import jax
jax.config.update("jax_enable_x64", True)

from jax import Array, vmap#, jit
import jax.numpy as jnp
import jax.random as jrnd

from flax import linen as nn

from functools import partial

from jax.nn import log_softmax


class particle_conservation(nn.Module):
    """
    Wrapper module for particle conservation (bosons) for autoregressive networks which have an elementwise execution.

    Initialization arguments:
        * ``net``: Flax module defining the plain ansatz.
        * ``Q``: Fixed particle number 

    """
    #orbit: LatticeSymmetry
    net: callable
    Q: int
    
    
    #avgFun: callable = avgFun_Coefficients_Exp
    def setup(self):
        self.is_particle = True
        self.L = self.net.L
        self.LocalHilDim = self.net.LocalHilDim
        self.must_mask = 2 * jnp.tril(jnp.ones((self.LocalHilDim,self.LocalHilDim)),k=-1)
        self.can_mask = jnp.flip(self.must_mask)
        self.max_particles = jnp.pad((self.LocalHilDim-1)*jnp.arange(1,self.L+1)[::-1],(0,1))[1::]
        self.logProbFactor = self.net.logProbFactor
    def __post_init__(self):

        super().__post_init__()
    
    def __call__(self,*args,**kwargs):
        
        if "output_state" in kwargs:
            cumsum_left = self.Q - kwargs["cumsum"]
            position = kwargs["position"]
            del kwargs["position"],kwargs["cumsum"]
            x, state = self.net(*args,**kwargs)
            must_give = nn.relu(cumsum_left-self.max_particles[position])
            can_give = jnp.minimum(cumsum_left, self.LocalHilDim-1)
            mask = (self.must_mask[must_give] + 
                        self.can_mask[can_give.astype(int)]) 
            
            ##############################################
            x = x - mask ** jnp.inf
            
            return x, state

        else: 
            #flagOutput = False #kwargs["output_state"]
            kwargs["output_state"] =True
            s = args[0]
            y = jnp.pad(s[:-1],(1,0),mode='constant',constant_values=0)
            cum_sum = self.Q - jnp.cumsum(y)
            #x, state = self.net(*args,output_state=True,**kwargs)
            x, state = self.net(*args,**kwargs)
            
            must_give = nn.relu(cum_sum-self.max_particles)
            can_give = jnp.minimum(cum_sum, self.LocalHilDim-1)
            mask = (self.must_mask[must_give] + 
                        self.can_mask[can_give.astype(int)]) 
            x = x - mask ** jnp.inf
            x = log_softmax(x)
            x *= self.logProbFactor
            return (jnp.take_along_axis(x, jnp.expand_dims(s, -1), axis=-1)
                                    .sum(axis=-2)
                                    .squeeze(-1))
    
    def _apply_fun(self, *args,**kwargs):
        return self.net.apply(*args,**kwargs)

    def sample(self,numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = jrnd.split(key, self.L)
            logits, carry = self.net(jnp.zeros(1,dtype=int),block_states = None, output_state=True)
            must_give = nn.relu(self.Q-self.max_particles[0])
            can_give = jnp.minimum(self.Q, self.LocalHilDim-1)
            mask = (self.must_mask[must_give] + 
                        self.can_mask[can_give.astype(int)]) 
            
            ##############################################
            logits = logits - mask ** jnp.inf
            choice = jrnd.categorical(key[0], logits.ravel()) # abide by the indexing convention and apply -1
            s_cumsum = self.Q - choice # create cumsum of the quantum number
            
            _, s = self._scanning_fn((jnp.expand_dims(choice,0),carry,s_cumsum),(key[1:],jnp.arange(1,self.L)))
            return jnp.concatenate([jnp.expand_dims(choice,0),s])

        # get the samples
        keys = jrnd.split(key, numSamples)
        samples = vmap(generate_sample)(keys)
        # return to the spinless representation
        return samples

    @partial(nn.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry, key):
        logits, next_states = self.net(carry[0],block_states = carry[1], output_state=True)
        
        must_give = nn.relu(carry[2]-self.max_particles[key[1]])
        can_give = jnp.minimum(carry[2], self.LocalHilDim-1)
        mask = (self.must_mask[must_give] + 
                    self.can_mask[can_give.astype(int)]) 
        
        ##############################################
        logits = logits - mask ** jnp.inf
        choice = jrnd.categorical(key[0], logits.ravel().real) # abide by the indexing convention
        ##############################################
        s_cumsum = carry[2] - choice
        return (jnp.expand_dims(choice,0), next_states, s_cumsum), choice