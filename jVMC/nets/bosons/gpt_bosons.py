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
    log_softmax,
    softmax,
    make_causal_mask,
    scan,
)
from jax import Array, vmap
from jax.config import config  # type: ignore
from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros
from jax.random import KeyArray, categorical, split
from jVMC.global_defs import tReal
import jVMC


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
        
        
        y = Sequential(
            [
                #Dense(4*self.embeddingDim, param_dtype=self.paramDType),
                Dense(4*self.embeddingDim, param_dtype=self.paramDType),
                #relu,
                gelu,
                Dense(self.embeddingDim, param_dtype=self.paramDType),
                gelu,
            ]
        )(x)
        x = x + y
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
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe.T[:, :x.shape[1]].squeeze()
        return x

class GPT(Module):
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

    L: int
    N: int
    lDim: int
    embeddingDim: int
    depth: int
    nHeads: int
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64

    @compact
    def __call__(self, s: Array, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function.
        """
        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        if not s.shape[-1] == self.L:
            raise ValueError(
                "Input length should be equal to context length, L."
            )
        #y = jax.nn.one_hot(s,self.N+1,dtype=int)
        sIn = jnp.roll(s,1).at[0].set(0)
        y = Embed(self.lDim, self.embeddingDim, param_dtype=self.paramDType)(sIn)
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
        p = PositionalEncoding(self.L,self.embeddingDim)
        y = p(y) 
        y = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.nHeads, self.paramDType
                )
                for _ in range(self.depth)
            ]
        )(y)
        y = Dense(self.lDim, param_dtype=self.paramDType)(y)
        #jax.debug.print("{y}",y=y)
        
        testVal = -jnp.inf # value for the unphysical particle numbers
        maskPhysical = self.N - jnp.roll(jnp.cumsum((s+abs(s))/2),1).astype(int) # particle left to distribute  
        maskPhysical = maskPhysical.at[0].set(self.N) # first site all particles allowed 
        
        yPhysical = jnp.where(jnp.arange(self.lDim)[None,:]<=maskPhysical[:,None], 0,testVal)
        
        yMaxLeft = jnp.arange(self.L-1,-1,-1)*(self.lDim-1)
        yPhysical2 = jnp.where(jnp.arange(self.lDim)[None,:]>=(-yMaxLeft[:,None]+maskPhysical[:,None]), 0,testVal)
        #jax.debug.print("{x}", x=yPhysical2)

        yPhysical += yPhysical2
        #yPhysical = yPhysical.at[-1].set(testVal).at[-1,maskPhysical[-1]].set(1)
        y = y + yPhysical
        y = jnp.nan_to_num(y,nan=testVal)

        y = log_softmax(y)* self.logProbFactor
        #return y 
        #jax.debug.print("{x}", x=jnp.exp(y))

        #a = take_along_axis((y), expand_dims(s, -1), axis=-1)
        #jax.debug.print("{x}\n {s} ",x=a,s=s)
        
        if returnLogAmp:
            
            return (
                #take_along_axis(log_softmax(y), expand_dims(s, -1), axis=-1)
                take_along_axis((y), expand_dims(s, -1), axis=-1)
                .sum(axis=-2)
                .squeeze(-1)
                
            )
        
        return y

    def sample(self, numSamples: int, key: KeyArray) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """

        def generate_sample(key):
            keys = split(key, self.L)
            s = full((self.L,), -1, self.spinDType)
            s, _ = self._scanning_fn(s, (keys, arange(self.L)))
            #keyShift,keyDummy = split(keys[0],2)
            #shift = jax.random.randint(keyShift,(1,),0,self.L) # not okay
            #return jnp.roll(s,shift)
            return s
        keys = split(key, numSamples)
        res =  vmap(generate_sample)(keys)
        return res 
    @partial(scan, variable_broadcast="params", split_rngs={"params": False})
    def _scanning_fn(
        self, s: Array, x: Tuple[KeyArray, Array]
    ) -> Tuple[Array, None]:
        logits = self(s, False)
        choice = categorical(x[0], logits[x[1]]/self.logProbFactor)
        
        return s.at[x[1]].set(choice), None


#maskPhysical = jax.numpy.cumsum(s)

#