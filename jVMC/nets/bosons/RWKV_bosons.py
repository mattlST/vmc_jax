from functools import partial
from itertools import repeat
from typing import Any, List, Optional, Tuple

import flax
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, vmap, jit
import jax.random as jrnd
# from jVMC.global_defs import tReal
tReal = jnp.float64

def initialize_to_value(x, dtype):
    """
    makes an initializer function that ignores the given PRNGKey
    and always returns the given value
    """
    return lambda _: x.astype(dtype)


class TimeMixing(nn.Module):
    layer_depth: int = 1
    num_layers: int = 1
    embedding_size: int = 2
    dtype: type = tReal

    def setup(self):
        #####################################################
        # initialize the \mu for time mixing
        #####################################################
        # goes from 0 to 1 along layer depth
        ratio_0_to_1 = self.layer_depth / (self.num_layers - 1)
        # goes from 1 to (almost) 0 along layer depth
        ratio_1_to_almost_0 = 1.0 - (self.layer_depth / self.num_layers)
        zigzag = .5 * (jnp.arange(1, self.embedding_size+1) % 3 - 1)
        time_first = jnp.full(self.embedding_size, jnp.log(.3)) + zigzag
        self.time_first = self.param(
            'time_first', initialize_to_value(time_first, self.dtype))

        h = jnp.arange(0, self.embedding_size)
        # the numbers used here were found to work well from experiments
        time_decay = -5 + 8 * (h / (self.embedding_size - 1)
                               ) ** (.7 + 1.3 * ratio_0_to_1)

        self.time_decay = self.param(
            'time_decay', initialize_to_value(time_decay,  self.dtype))

        x = (jnp.arange(self.embedding_size) /
             self.embedding_size)
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_v = time_mix_k + .3 * ratio_0_to_1
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)
        self.time_mix_k = self.param(
            'time_mix_k', initialize_to_value(time_mix_k, self.dtype))
        self.time_mix_v = self.param(
            'time_mix_v', initialize_to_value(time_mix_v, self.dtype))
        self.time_mix_r = self.param(
            'time_mix_r', initialize_to_value(time_mix_r, self.dtype))

        #####################################################
        # set up the linear RKV layers and the output layer
        #####################################################
        #self.layernorm = nn.LayerNorm(epsilon=1e-5, param_dtype=self.dtype)
        self.key = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.value = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.receptance = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype)
        self.output = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype)

    def __call__(self, x, time_mix_state):
        # get the state from the previous time step
        sx, aa, bb, pp = time_mix_state # sx is the previous time_mix_state
        xx = x
        #xx = self.layernorm(x)
        sx = jnp.concatenate((jnp.expand_dims(sx, 0), xx[:-1, :]))
        kx = xx * self.time_mix_k + sx * (1 - self.time_mix_k)
        vx = xx * self.time_mix_v + sx * (1 - self.time_mix_v)
        rx = xx * self.time_mix_r + sx * (1 - self.time_mix_r)
        # apply the gating mechanism
        r = nn.sigmoid(self.receptance(rx)) # does not go into the scan
        # compute the key and value
        k = self.key(kx)
        v = self.value(vx)
        # T = x.shape[0]

        # perform the WKV activation
        def step(state, kv):
            ############
            # KV: key and value pair
            ###########
            (aa, bb, pp), (kk, vv) = state, kv
            ww = self.time_first + kk
            p = jnp.maximum(pp, ww)
            e1 = jnp.exp(pp - p)
            e2 = jnp.exp(ww - p)
            out = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).astype(dtype=r.dtype)

            # regularize the exponential moving attention
            ww = -jnp.exp(self.time_decay) + pp
            p = jnp.maximum(ww, kk)
            e1 = jnp.exp(ww - p)
            e2 = jnp.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
            return (aa, bb, pp), out

        (aa, bb, pp), sx = jax.lax.scan(step, (aa, bb, pp), (k, v))
        # compute the output and recombine with the input
        out = x + self.output(r * sx)

        # (xx[-1,:], aa, bb, pp) is the next time_mix_state
        return out, (xx[-1, :], aa, bb, pp)


class ChannelMixing(nn.Module):
    layer_depth: int = 1
    num_layers: int = 1
    embedding_size: int = 2
    dtype: type = tReal

    def setup(self):
        #####################################################
        # set up the linear RKV layers and the output layer
        #####################################################
        hidden_size = 4 * self.embedding_size
        #self.layernorm = nn.LayerNorm(epsilon=1e-5,  param_dtype=self.dtype,name="LayerNormP")
        self.key = nn.Dense(hidden_size, use_bias=False, param_dtype=self.dtype,name="KeyP")
        self.receptance = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype,name="ReceptanceP")
        self.value = nn.Dense(self.embedding_size, use_bias=False, param_dtype=self.dtype,name="ValueP")

        #####################################################
        # initialize the \mu for time mixing
        #####################################################
        x = (jnp.arange(self.embedding_size) /
             self.embedding_size)

        ratio_1_to_almost_0 = 1.0 - (self.layer_depth / self.num_layers)
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)
        self.time_mix_k = self.param(
            'time_mix_k', initialize_to_value(time_mix_k, self.dtype))
        self.time_mix_r = self.param(
            'time_mix_r', initialize_to_value(time_mix_r, self.dtype))

    def __call__(self, x, channel_mix_state):
        xx = x #self.layernorm(x)
        sx = jnp.concatenate(
            (jnp.expand_dims(channel_mix_state, 0), xx[:-1, :]))
        xk = xx * self.time_mix_k + sx * (1 - self.time_mix_k)
        xr = xx * self.time_mix_r + sx * (1 - self.time_mix_r)
        # sigmoid activation
        r = nn.sigmoid(self.receptance(xr))
        # square ReLU activation
        k = jnp.square(nn.relu(self.key(xk)))
        # compute the output and recombine with the input
        out = x + r * self.value(k)

        return out, xx[-1, :]

def empty_state(embedding_size):
    "returns an empty block_state for a given embedding size"
    zeros = jnp.zeros(embedding_size)
    min_values = jnp.full(embedding_size, -jnp.inf)
    time_mix_state = (zeros, zeros, zeros, min_values)
    channel_mix_state = zeros
    return time_mix_state, channel_mix_state

class RWKVBlock(nn.Module):
    # Block
    layer_num: int = 1
    dtype: type = tReal
    # time/channel mixing
    num_layers: int = 1
    embedding_size: int =2

    def setup(self):
        self.time_mix = TimeMixing(layer_depth=self.layer_num,
                                   embedding_size=self.embedding_size,
                                   num_layers=self.num_layers,
                                   dtype=self.dtype)
        self.channel_mix = ChannelMixing(layer_depth=self.layer_num,
                                   embedding_size=self.embedding_size,
                                   num_layers=self.num_layers,
                                   dtype=self.dtype)

    def __call__(self, x, block_state):
        """
        Takes the embedding from the previous layer, and the `block_state`
        from the previous time_step.

        `block_state` is a tuple of `time_mix_state` and `channel_mix_state`,
        which are used as inputs to the block's `time_mix` and `channel_mix`
        respectively.
        """
        if block_state is None:
            block_state = empty_state(self.embedding_size)

        time_mix_state, channel_mix_state = block_state
        x, time_mix_state = self.time_mix(x, time_mix_state)
        x, channel_mix_state = self.channel_mix(x, channel_mix_state)
        return x, (time_mix_state, channel_mix_state)

class RWKV(nn.Module):
    # Main
    L: int = 1 # system size
    Q: int = None # charge quantum number --> particle number 
    M: int = None # magnetization quantum number
    LocalHilDim: int = 4 # local hilbert space dimension for spinful fermions
    dtype: type = tReal
    order: int = 1
    # time/channel mixing
    num_layers: int = 1
    embedding_size: int = 2
    # prob correction
    logProbFactor: float = 0.5

    def setup(self):
        # fermion maps 
        #self.sfm = spinful_fermion_map(order=self.order)
        #self.sfu = spinful_fermion_unmap(order=self.order)
        # network layers
        self.embed = nn.Embed(self.LocalHilDim,
                              self.embedding_size,
                              param_dtype=self.dtype)
        #self.input_layernorm = nn.LayerNorm(epsilon=1e-5,name="InputLayerNorm",param_dtype=self.dtype)
        self.blocks = [
                        RWKVBlock(layer_num=i,
                              embedding_size=self.embedding_size,
                              num_layers=self.num_layers,
                              dtype=self.dtype)
                        for i in range(self.num_layers)
                       ]
        #self.output_layernorm = nn.LayerNorm(epsilon=1e-5,name="OutputLayerNorm", param_dtype=self.dtype)
        #self.neck = nn.Dense(self.embedding_size * self.LocalHilDim, use_bias=True,name="Neck", param_dtype=self.dtype)
        self.neck = nn.Dense(self.embedding_size, use_bias=True,name="Neck", param_dtype=self.dtype)
        self.head = nn.Dense(self.LocalHilDim, use_bias=False,name="Head", param_dtype=self.dtype)
        # spin masks
        #self.must_mask = jnp.array([[0.,0.,0.,0.],[0.,2.,0.,0.],[2.,2.,2.,0.]])
        #self.can_mask = jnp.array([[2.,0.,2.,2.],[0.,0.,0.,2.],[0.,0.,0.,0.]])

        self.must_mask = 2 * jnp.tril(jnp.ones((self.LocalHilDim,self.LocalHilDim)),k=-1)
        self.can_mask = jnp.flip(self.must_mask)
        
        
        self.max_particles = jnp.pad((self.LocalHilDim-1)*jnp.arange(1,self.L+1)[::-1],(0,1))[1::]
        #self.max_particles = jnp.pad((2*jnp.arange(1,self.L+1)[::-1],(0,1))[1::]
        #jax.debug.print("x {x}", x= self.max_particles.shape)
    def __call__(self, s: Array, block_states: Array = None, output_state: bool = False) -> Array:
        # the helper method allows to use nn.jit with static_argnames
        return self.forward_with_state(s, block_states, output_state)

    @partial(nn.jit, static_argnums=3)
    def forward_with_state(self, s: Array, block_states: Array = None,  output_state: bool = False) -> Array:

        # embed the input and apply the input layer norm
        if output_state:
            x = self.embed(s)
        else:
        #    # apply the fermion map
        #    s = self.sfm(s)
            # ah  okay sets s= as the input of the network [0,s[:-1]]
            x = jnp.pad(s[:-1],(1,0))
            x = self.embed(x)
        #x = self.embed(s) ## not sure if I need that if condition
        #x = self.input_layernorm(x)

        next_states = []
        if block_states is None:
            block_states = repeat(None)
        for block, state in zip(self.blocks, block_states):
            x, new_state = block(x, state)
            if output_state:
                next_states.append(new_state)

        #x = self.output_layernorm(x)
        # the neck precedes the head
        x = nn.gelu(self.neck(x))
        # the head tops the neck
        x = self.head(x)

        # return here for RNN mode
        if output_state:
            return x, next_states
        # for prediction mode the ouput is returned after soft_max activation
        unphyiscal = 0
        if self.Q is not None:
            # computing the quantum number
            #sQ = jnp.sum(abs(s)) # s quantum number
            Q_diff = self.Q-jnp.sum(s)
            #####################################
            # setting unphysical states to zeros
            unphyiscal = 2 * abs(Q_diff)
            #s_cumsum = self.Q - jnp.cumsum(abs(s))
            s_cumsum = self.Q - jnp.cumsum(s)

            #jax.debug.print("x {x}", x=s_cumsum)
            #####################################
            # manually enforce the right probabilites
            must_give = nn.relu(jnp.roll(s_cumsum,1)-self.max_particles)
            # trivial deterministic mask
            can_give = jnp.minimum(s_cumsum,(self.LocalHilDim-1)*jnp.ones(s_cumsum.shape))
            can_give = jnp.pad(can_give,(1,0), 'constant', constant_values=self.LocalHilDim-1)[:-1]
            # mask for the last site
            # NOTE: if `s` is unphysical no infinities can be in the mask
            """
            jax.debug.print("must_give {x}\n can_give {y}",x=must_give,y=can_give)
            jax.debug.print("must_mask {x}\n can_mask {y}",x=self.must_mask,y=self.can_mask)

            jax.debug.print("must {x}\n can {y}",x=self.must_mask[must_give],y=self.can_mask[can_give.astype(int)])
            jax.debug.print("must shape {x}\n can shape {y}",x=self.must_mask[must_give].shape,y=self.can_mask[can_give.astype(int)].shape)

            jax.debug.print("sum must can {x}",x=self.must_mask[must_give] + self.can_mask[can_give.astype(int)]) 
            jax.debug.print("unphyiscal {x}", x=1 - jnp.sign(unphyiscal))
            jax.debug.print("x {x}",x=x)
            """
            
            mask = (self.must_mask[must_give] + 
                    self.can_mask[can_give.astype(int)]) * (1 - jnp.sign(unphyiscal)) 
            # applying the mask and setting the unphysical tree leafs to -inf
            x = x - mask ** jnp.inf
            #jax.debug.print("x {x}",x=x)

        #if self.M is not None:
        #    m = (s % 2) * jnp.sign(s)
        #    # computing the quantum number
        #    sM = jnp.sum(m) # s quantum number
        #    Q_diff = self.M-sM

        #####################################
        x = nn.log_softmax(x) * self.logProbFactor
        # the log-probs according the state
        return (jnp.take_along_axis(x, jnp.expand_dims(
                                s, # here we shift the state by one to match the indexing
                                axis=-1), axis=-1)
                                .sum(axis=-2)
                                .squeeze(-1)) - (unphyiscal**jnp.inf)

    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = jrnd.split(key, self.L)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True)

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
        logits, next_states = self(carry[0],block_states = carry[1], output_state=True)
        #jax.debug.print("logits {x}\n carry {y}",x=logits,y=carry)

        ##############################################
        # mask logits
        # number of particles that mus assigned
        must_give = nn.relu(carry[2]-self.max_particles[key[1]])
        # number of particles that can be assigned
        can_give = jnp.minimum(carry[2], self.LocalHilDim-1)
        # compute mask
        mask = (self.must_mask[must_give] + 
                    self.can_mask[can_give.astype(int)]) 
        ##############################################
        logits = logits - mask ** jnp.inf
        choice = jrnd.categorical(key[0], logits.ravel().real) # abide by the indexing convention
        ##############################################
        s_cumsum = carry[2] - choice
        return (jnp.expand_dims(choice,0), next_states, s_cumsum), choice