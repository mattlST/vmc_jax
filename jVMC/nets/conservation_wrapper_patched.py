import jax
jax.config.update("jax_enable_x64", True)

from jax import Array, vmap#, jit
import jax.numpy as jnp
import jax.random as jrnd

from flax import linen as nn

from functools import partial

from jax.nn import log_softmax


class particle_conservation_patched(nn.Module):
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
        self.patch_size = self.net.patch_size
        self.LHilDim = self.net.LHilDim
        self.LocalHilDim = self.net.LocalHilDim
        self.patch_states = self.net.patch_states 
        self.PL = self.net.PL
        index_array = self.net.LHilDim**(jnp.arange(self.patch_size)[::-1])
        self.index_map = jax.vmap(lambda s: index_array.dot(s))
        
        self.maxParticleP1 = (self.LHilDim-1) * self.patch_size + 1
        self.maxParticleSite =  (self.LHilDim-1) * self.patch_size 
        
        particles = jnp.sum(self.patch_states,axis=1)
        self.must_mask = 2*jnp.array([particles<j for j in range(self.maxParticleP1)])

        # 2 * jnp.tril(jnp.ones((self.LocalHilDim,self.LocalHilDim)),k=-1)
        self.can_mask = jnp.flip(self.must_mask)

        self.max_particles = jnp.pad(self.maxParticleSite*jnp.arange(1,self.PL+1)[::-1],(0,1))[1::]
        self.logProbFactor = self.net.logProbFactor
    def __post_init__(self):

        super().__post_init__()
    
    def __call__(self,*args,**kwargs):
        
        if "output_state" in kwargs:
            #jax.debug.print("eeee")
            cumsum_left = self.Q - kwargs["cumsum"]
            position = kwargs["position"]
            must_give = nn.relu(cumsum_left-self.max_particles[position])
            can_give = jnp.minimum(cumsum_left, self.maxParticleSite)
            mask = (self.must_mask[must_give] + 
                        self.can_mask[can_give.astype(int)]) 
            del kwargs["position"],kwargs["cumsum"]
            x, state = self.net(*args,**kwargs)
            ##############################################
            x = x - mask ** jnp.inf
            x = log_softmax(x)
            return x, state

        else: 
            #flagOutput = False #kwargs["output_state"]
            #jax.debug.print('ddd')
            kwargs["output_state"] =True
            s = args[0]
            s = self.index_map(s.reshape(self.PL,self.patch_size))

            y = jnp.pad(s[:-1],(1,0),mode='constant',constant_values=0)
            #jax.debug.print("y: {y}", y=y)
            #jax.debug.print("yys: {ys}", ys=jnp.take_along_axis(self.patch_states,y[:,None],axis=0))
            ys = jnp.take_along_axis(self.patch_states,y[:,None],axis=0).sum(axis=(1)).flatten()
            #jax.debug.print("ys: {ys}", ys=ys)
            
            cum_sum = self.Q - jnp.cumsum(ys)
            #x, state = self.net(*args,output_state=True,**kwargs)
            if ((self.net.__name__ == "GPT") or (self.net.__name__ == "GPT_patched")): # not happy
                x = self.net.call_all(y,output_state=True)

            else:
                x, state = self.net(y,**kwargs)
                
            must_give = nn.relu(cum_sum-self.max_particles)
            can_give = jnp.minimum(cum_sum, self.maxParticleSite)
            mask = (self.must_mask[must_give] + 
                        self.can_mask[can_give.astype(int)]) 
            x = x - mask ** jnp.inf
            x = log_softmax(x)*self.logProbFactor
            #x *= self.logProbFactor    
            if hasattr(self.net,"flag_phase"):
                if self.net.flag_phase:
                    y = self.net.embed(y)
                    if hasattr(self.net.PhaseAttention,"__len__"):
                        # compute the phase in the auotregressive style
                        # phase = nn.gelu(self.PhaseNeck(y[-1]))
                        phase = self.net.PhaseAttention[0](y)
                        for i in range(1,self.net.num_layers_phase):
                            phase = self.net.PhaseAttention[i](phase) 
                    else:
                        phase = self.net.PhaseAttention(y)
                    phase = self.net.PhaseHead(phase)
                    # the log-probs according the state
                    return (jnp.take_along_axis(x, jnp.expand_dims(s, -1), axis=-1)
                                            .sum(axis=-2)
                                            .squeeze(-1) 
                                        #    +1.j * phase.squeeze(-1) )
                            + 1.j * jnp.take_along_axis(phase, jnp.expand_dims(s, -1), axis=-1)
                                            .sum(axis=-2)
                                            .squeeze(-1) )
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
            key = jrnd.split(key, self.PL)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True,cumsum=0,position=0)
            #logits, carry = self.net(jnp.zeros(1,dtype=int),block_states = None, output_state=True)
            #must_give = nn.relu(self.Q-self.max_particles[0])
            #can_give = jnp.minimum(self.Q, self.LocalHilDim-1)
            #mask = (self.must_mask[must_give] + 
            #            self.can_mask[can_give.astype(int)]) 
            
            ##############################################
            #logits = logits - mask ** jnp.inf
            choice = jrnd.categorical(key[0], logits.ravel()) # abide by the indexing convention and apply -1
            #s_cumsum = self.Q - choice # create cumsum of the quantum number
            cumsum = self.patch_states[choice].sum() # create cumsum of the quantum number
            
            #_, s = self._scanning_fn((jnp.expand_dims(choice,0),carry,s_cumsum),(key[1:],jnp.arange(1,self.L)))
            _, s = self._scanning_fn((jnp.expand_dims(choice,0),carry,cumsum),(key[1:],jnp.arange(1,self.PL)))
            
            ##
            state = jnp.concatenate([jnp.expand_dims(choice,0),s])
            return jnp.take_along_axis(self.patch_states,state[:,None],axis=0).flatten()

            #return jnp.concatenate([jnp.expand_dims(choice,0),s])

        # get the samples
        keys = jrnd.split(key, numSamples)
        samples = vmap(generate_sample)(keys)
        # return to the spinless representation
        return samples

    @partial(nn.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry, key):
        logits, next_states = self(carry[0],block_states = carry[1], output_state=True,cumsum=carry[2],position=key[1])
        
        #must_give = nn.relu(carry[2]-self.max_particles[key[1]])
        #can_give = jnp.minimum(carry[2], self.LocalHilDim-1)
        #mask = (self.must_mask[must_give] + 
        #            self.can_mask[can_give.astype(int)]) 
        
        ##############################################
        #logits = logits - mask ** jnp.inf
        choice = jrnd.categorical(key[0], logits.ravel().real) # abide by the indexing convention
        ##############################################
        s_cumsum = carry[2] + self.patch_states[choice].sum()
        #return (jnp.expand_dims(choice,0), next_states, s_cumsum), choice
        return (jnp.expand_dims(choice,0), next_states,s_cumsum), choice