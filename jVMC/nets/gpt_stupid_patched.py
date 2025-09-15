import numpy as np 
from functools import partial
from typing import Tuple
import itertools
import math
import jax 
import jax.numpy as jnp 
import flax.linen as nn
from flax.linen import (
    Dense,
    Embed,
    LayerNorm,
    Module,
    MultiHeadDotProductAttention,
    Sequential,
    compact,
    gelu,
    elu,
    relu,
    glu,
    PReLU,
    log_softmax,
    softmax,
    make_causal_mask,
    scan,
)
from jax import Array, vmap
#from jax.config import config  # type: ignore
from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros
from jax.random import categorical, split
from jVMC.global_defs import tReal
import jVMC
from itertools import product

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.nets.initializers import init_fn_args

from functools import partial

import jVMC.nets.initializers

# Transformer

class _TransformerBlock(Module):
    """The transformer decoder block."""
    
    embeddingDim: int
    nHeads: int
    paramDType: type = tReal

    @compact
    def __call__(self, x: Array) -> Array:
        x = x + MultiHeadDotProductAttention(
            self.nHeads, param_dtype=self.paramDType
        )(
            x,
            x,
            mask=make_causal_mask(
                zeros((x.shape[-2]), self.paramDType), dtype=self.paramDType
            ),
        )
        #jax.debug.print("{x}", x=x)
        #x = LayerNorm(param_dtype=self.paramDType)(x)
        
        
        # y = Sequential(
        #     [
        #         #Dense(4*self.embeddingDim, param_dtype=self.paramDType),
        #         Dense(4*self.embeddingDim, param_dtype=self.paramDType),
        #         #relu,
        #         gelu,
        #         #glu,
        #         #PReLU(param_dtype=self.paramDType),
        #         Dense(self.embeddingDim, param_dtype=self.paramDType),
        #     ]
        # )(x)
        # x = x + y
        x = x+ Sequential(
            [
                #Dense(4*self.embeddingDim, param_dtype=self.paramDType),
                Dense(4*self.embeddingDim, param_dtype=self.paramDType),
                #relu,
                gelu,
                #glu,
                #PReLU(param_dtype=self.paramDType),
                Dense(self.embeddingDim, param_dtype=self.paramDType),
            ]
        )(x)
        #jax.debug.print("x {x} \ny {y}\n", x=x,y=y)
        #x = LayerNorm(param_dtype=self.paramDType)(x)
        return x


class PositionalEncoding(nn.Module):
    d_model : int  # Hidden dimensionality of the input.
    max_len : int  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, self.d_model+(1 if self.d_model%2==1 else 0), 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)[:,:self.d_model//2]
        #pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe.T[:, :x.shape[1]].squeeze()
        return x

class GPT_patched(Module):
    """GPT model for autoregressive decoding of neural quantum states.

    This model outputs the log amplitude of a wave function which in turn is
    a log probability density. It contains a ``sample`` method that peforms
    autorgressive sampling.

    Initialization arguments:
        * ``L``: Length of the spin chain.
        * ``embeddingDim``: Embedding dimension.
        * ``depth``: Number of transformer blocks.
        * ``nHeads``: Number of attention heads.
        * ``logProbFactor``: Factor defining how output and associated sample
                probability are related. 0.5 for pure states and 1.0 for POVMs
                (default: 0.5).
        * ``paramDType``: Data type of the model parameters
                (default: ``jVMC.global_defs.tReal``).
        * ``spinDType``: Data type of the spin configurations
                (default: ``jax.numpy.int64``).
    """

    L: int = 1 # system size
    LHilDim: int = 2
    patch_size: int = 1
    embeddingDim: int = 4
    depth: int = 1
    nHeads: int = 1
    hiddenDim: int = 16
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64
    __name__: str = "GPT"
    def setup(self):
        self.LocalHilDim = self.LHilDim ** self.patch_size
        self.lDim = self.LocalHilDim
        self.ldim = self.LocalHilDim
        if self.L % self.patch_size != 0:
            raise ValueError("The system size must be divisible by the patch size")
        self.patch_states = jnp.array(list(product(range(self.LHilDim),repeat=self.patch_size)))
        self.ldim =self.LocalHilDim
        self.lDim = self.LocalHilDim

        self.PL = self.L // self.patch_size
        index_array = self.LHilDim**(jnp.arange(self.patch_size)[::-1])
        self.index_map = jax.vmap(lambda s: index_array.dot(s))
        
        self.posEnc = PositionalEncoding(self.PL,self.embeddingDim)
        
        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )

    def __call__(self, s : Array, block_states=None, output_state=False) -> Array:
        #jax.debug.print("s {s}",s=s)

        if output_state==False:
            s = self.index_map(s.reshape(self.PL,self.patch_size))
            return self.call_all(s,output_state= False)

        if block_states is None:
            s_call = jnp.full(self.PL,0,dtype=int)
            index = 0
        else:
            s_call, index = block_states
            #jax.debug.print("s_call {x}, index {y}",x=s_call,y=index)
            s_call = s_call.at[index].set(s[0])
        next_state = (s_call,index+1)
        #jax.debug.print("index: {x}",x=index)
        #jax.debug.print("state: {x}",x=s_call)
        logits = self.call_all(s_call,output_state=True)
        return logits[index],next_state
        
    @compact
    def call_all(self, s: Array, output_state: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function.
        """
        
        #y = jax.nn.one_hot(s,self.N+1,dtype=int)
        if output_state==False:
            y = jnp.roll(s,1).at[0].set(0)
        else:
            y = s
        #y = s# jnp.pad(s,(1,0),mode='constant',constant_values=-1)

        y = Embed(self.lDim, self.embeddingDim, param_dtype=self.paramDType)(y)
        #y = LayerNorm(param_dtype=self.paramDType)(y)

        #y = jnp.nan_to_num(y)
        #y = Embed(self.N+1, self.embeddingDim, param_dtype=self.paramDType)(s)
        #p = self.variable(
        #    "params",
        #    "positional_embeddings",
        #    zeros,
        #    (self.L, self.embeddingDim),
        #    self.paramDType,
        #).value
        #y = y+p
        #jax.debug.print("{y},{p}",y=y.shape,p=p.shape)
        y = self.posEnc(y) 
        y = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.nHeads, self.paramDType
                )
                for _ in range(self.depth)
                
            ]
        )(y)
        y = Dense(self.hiddenDim, param_dtype=self.paramDType)(y)
        y = glu(y)
        #y = PReLU(param_dtype=self.paramDType)(y)
        y = Dense(self.lDim, param_dtype=self.paramDType)(y)
        y = LayerNorm(param_dtype=self.paramDType)(y)
        y = log_softmax(y)
        #jax.debug.print("logits all: {x}",x=y)
        #jax.debug.print("shape {x}",x=y.shape)
        if output_state==False:
            
            return (
                #take_along_axis(log_softmax(y), expand_dims(s, -1), axis=-1)
                take_along_axis((y), expand_dims(s, -1), axis=-1)
                .sum(axis=-2)
                .squeeze(-1)* self.logProbFactor)
                
        
        return y


    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = split(key, self.PL)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True)
            choice = categorical(key[0], logits.ravel().real)
            x, s = self._scanning_fn((jnp.expand_dims(choice,0),carry),key[1:])
            state =  jnp.concatenate([jnp.expand_dims(choice,0),s])
            return jnp.take_along_axis(self.patch_states,state[:,None],axis=0).flatten()

        # get the samples
        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry: Tuple[Array, Array, float], key) -> Tuple[Array,Array, float]:
        logits, next_states = self(carry[0],block_states = carry[1], output_state=True)
        choice = categorical(key, logits.ravel().real)
        return (jnp.expand_dims(choice,0), next_states), choice
