"""Simple GPT model for autoregressive encoding of quantum states."""

from functools import partial
from typing import Tuple

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
    log_softmax,
    make_causal_mask,
    scan,
)
import jax
from jax import Array, vmap, debug, jit, config
#from jax.config import config  # type: ignore
import jax.numpy as jnp
from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros, roll, log, ones, pi, sin, log
from jax.nn import elu
from jax.random import categorical, split
from jVMC.global_defs import tReal

from itertools import product

config.update("jax_enable_x64", True)


class _TransformerBlock(Module):
    """The transformer decoder block."""

    embeddingDim: int
    hiddenDim: int
    nHeads: int
    init_variance: float = 0.1
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
        #x = LayerNorm(param_dtype=self.paramDType)(x)
        x = x + Sequential(
            [
                # the hidden layer and the embedding layer
                Dense(self.hiddenDim, param_dtype=self.paramDType,
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal")),
                gelu,
                Dense(self.embeddingDim, param_dtype=self.paramDType,
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal")),
            ]
        )(x)
        #x = LayerNorm(param_dtype=self.paramDType)(x)
        return x


class Transformer(Module):
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
    embeddingDim: int
    hiddenDim: int
    depth: int
    nHeads: int
    temperature: float = 0.
    # patch size
    patch_size: int = 1
    # one hot embedding
    one_hot: bool = False
    lin_out: bool = True
    # init variance
    init_variance: float = 0.1
    # some stuff
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64
    # jbr
    LHilDim: int = 2

    def setup(self):
        # set up patching
        if self.L % self.patch_size != 0:
            raise ValueError("The system size must be divisible by the patch size")
        self.patch_states = jnp.array(list(product(range(self.LHilDim),repeat=self.patch_size)))
        self.LocalHilDim = self.LHilDim ** self.patch_size
        self.PL = self.L // self.patch_size
        index_array = self.LHilDim**(jnp.arange(self.patch_size)[::-1])
        self.index_map = jax.vmap(lambda s: index_array.dot(s))
        # select type of embedding
        if self.one_hot:
            self.embed = nn.Dense(self.embeddingDim, use_bias=False,name="Embedding", param_dtype=self.paramDType)
        else:
            self.embed = nn.Embed(self.LocalHilDim,
                               self.embeddingDim,
                               param_dtype=self.paramDType)
        # positional encoding
        self.p = self.variable(
            "params",
            "positional_embeddings",
            zeros, # jbr: this is the positional embedding, but it is all zeros
            (self.PL, self.embeddingDim),
            self.paramDType,
        ).value
        # sequential model
        self.sequential = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.hiddenDim, self.nHeads,self.init_variance,self.paramDType
                )
                for _ in range(self.depth)
            ]
        )
        # head 
        self.neck = nn.Dense(self.hiddenDim,name="Neck",
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)
        self.head = nn.Dense(self.LocalHilDim,name="Head", 
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)

    def __call__(self, s: Array, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function and the complex phase.
        """

        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        # debug.print("{x}",x=s.shape)
        if self.one_hot:
            if returnLogAmp: s = self.index_map(s.reshape(self.PL,self.patch_size))
            y = jnp.pad(s[:-1],(1,0),mode='constant',constant_values=0)
            y = jax.nn.one_hot(y,self.LocalHilDim,axis=-1) 
            y = self.embed(y)
        else:
            if returnLogAmp: s = self.index_map(s.reshape(self.PL,self.patch_size))
            y = jnp.pad(s[:-1],(1,0),mode='constant',constant_values=0)
            y = self.embed(y)
        # jbr: adding the positional encoding
        y = y + self.p # jbr: this makes no sense, p is all zeoros
        y = self.sequential(y)
        # continue with calulating the log amplitude
        # the neck precedes the head
        y = nn.gelu(self.neck(y))
        # the head tops the neck
        y = self.head(y)
        # output part
        if self.lin_out:
            # necessery: positive activation for the probabilities with some regularization
            y = nn.elu(y) + 1. + 1e-8
            # return here for RNN mode
            if returnLogAmp:
                # normalize the product probability distribution
                y = jnp.log(y/jnp.expand_dims(y.sum(axis=-1),axis=-1)) * self.logProbFactor
            else:
                # normalize the product probability distribution
                y = jnp.log(y/jnp.expand_dims(y.sum(axis=-1),axis=-1))
                return y
        else:
            if returnLogAmp:
                y = log_softmax(y, axis=-1) * self.logProbFactor
            else:
                return y
        # return the log amplitude
        return (
            (take_along_axis(y, expand_dims(s, -1), axis=-1)
            .sum(axis=-2)
            .squeeze(-1))
        )

    def sample(self, numSamples: int, key: Array) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """

        def generate_sample(key):
            # we only need 
            keys = split(key, self.PL)
            # jax.numpy.full(shape, fill_value, dtype=None, *, device=None)[source]
            s = full((self.PL,), -1, self.spinDType)
            # had to modify because of Jax version?
            s, _ = self._scanning_fn(s, (keys, arange(self.PL)))
            return jnp.take_along_axis(self.patch_states,s[:,None],axis=0).flatten()

        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, s: Array, x: Tuple[Array, Array]) -> Tuple[Array, None]:
        logits = self(s, False).real
        choice = categorical(x[0], logits[x[1]]/(1. + self.temperature))
        return s.at[x[1]].set(choice), None
    
class CpxTransformer(Module):
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
    embeddingDim: int
    hiddenDim: int
    depth: int
    nHeads: int
    temperature: float = 0.
    # patch size
    patch_size: int = 1
    # one hot embedding
    one_hot: bool = False
    lin_out: bool = True
    # init variance
    init_variance: float = 0.1
    # some stuff
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64
    # jbr
    LHilDim: int = 2

    def setup(self):
        # set up patching
        if self.L % self.patch_size != 0:
            raise ValueError("The system size must be divisible by the patch size")
        self.patch_states = jnp.array(list(product(range(self.LHilDim),repeat=self.patch_size)))
        self.LocalHilDim = self.LHilDim ** self.patch_size
        self.PL = self.L // self.patch_size
        index_array = self.LHilDim**(jnp.arange(self.patch_size)[::-1])
        self.index_map = jax.vmap(lambda s: index_array.dot(s))
        # select type of embedding
        if self.one_hot:
            self.embed = nn.Dense(self.embeddingDim, use_bias=False,name="Embedding", param_dtype=self.paramDType)
        else:
            self.embed = nn.Embed(self.LocalHilDim,
                               self.embeddingDim,
                               param_dtype=self.paramDType)
        # positional encoding
        self.p = self.variable(
            "params",
            "positional_embeddings",
            zeros, # jbr: this is the positional embedding, but it is all zeros
            (self.PL+1, self.embeddingDim),
            self.paramDType,
        ).value
        # sequential model
        self.sequential = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.hiddenDim, self.nHeads,self.init_variance,self.paramDType
                )
                for _ in range(self.depth)
            ]
        )
        # head 
        self.neck = nn.Dense(self.hiddenDim,name="Neck",
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)
        self.head = nn.Dense(self.LocalHilDim,name="Head", 
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)
        self.phase_neck = nn.Dense(self.hiddenDim,name="PhaseNeck",
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)
        self.phase_head = nn.Dense(1,name="PhaseHead", 
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)
    

    def __call__(self, s: Array, block_states: Tuple = None, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function and the complex phase.
        """
        if block_states is not None:
            s = jnp.concatenate(block_states[0],s,block_states[1])

        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        # debug.print("{x}",x=s.shape)
        if self.one_hot:
            if returnLogAmp: s = self.index_map(s.reshape(self.PL,self.patch_size))
            x = jnp.pad(s,(1,0),mode='constant',constant_values=0)
            x = jax.nn.one_hot(x,self.LocalHilDim,axis=-1) 
            x = self.embed(x)
        else:
            if returnLogAmp: s = self.index_map(s.reshape(self.PL,self.patch_size))
            x = jnp.pad(s,(1,0),mode='constant',constant_values=0)
            x = self.embed(x)
        # jbr: adding the positional encoding
        x = x + self.p # jbr: this makes no sense, p is all zeoros
        x = self.sequential(x)
        # continue with calulating the log amplitude
        # the neck precedes the head
        y = nn.gelu(self.neck(x[:-1]))
        # the head tops the neck
        y = self.head(y)
        # output part
        if self.lin_out:
            # necessery: positive activation for the probabilities with some regularization
            y = nn.elu(y) + 1. + 1e-8
            # return here for RNN mode
            if returnLogAmp:
                # normalize the product probability distribution
                y = jnp.log(y/jnp.expand_dims(y.sum(axis=-1),axis=-1)) * self.logProbFactor
            else:
                # normalize the product probability distribution
                y = jnp.log(y/jnp.expand_dims(y.sum(axis=-1),axis=-1))
                return y
        else:
            if returnLogAmp:
                y = log_softmax(y, axis=-1) * self.logProbFactor
            else:
                return y
        # compute the phase
        phase = nn.gelu(self.phase_neck(x[-1]))
        phase = self.phase_head(phase)
        # return the log amplitude
        return (
            (take_along_axis(y, expand_dims(s, -1), axis=-1)
            .sum(axis=-2)
            .squeeze(-1)) + 1j * phase.squeeze(-1)
        )

    def sample(self, numSamples: int, key: Array) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """

        def generate_sample(key):
            # we only need 
            keys = split(key, self.PL)
            # jax.numpy.full(shape, fill_value, dtype=None, *, device=None)[source]
            s = full((self.PL,), -1, self.spinDType)
            # had to modify because of Jax version?
            s, _ = self._scanning_fn(s, (keys, arange(self.PL)))
            return jnp.take_along_axis(self.patch_states,s[:,None],axis=0).flatten()

        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, s: Array, x: Tuple[Array, Array]) -> Tuple[Array, None]:
        logits = self(s, False).real
        choice = categorical(x[0], logits[x[1]]/(1. + self.temperature))
        return s.at[x[1]].set(choice), None

class PhaseTransformer(Module):
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
    embeddingDim: int
    hiddenDim: int
    depth: int
    nHeads: int
    temperature: float = 0.
    # patch size
    patch_size: int = 1
    # one hot embedding
    one_hot: bool = False
    lin_out: bool = True
    # init variance
    init_variance: float = 0.1
    # some stuff
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64
    # jbr
    LHilDim: int = 2

    def setup(self):
        # set up patching
        if self.L % self.patch_size != 0:
            raise ValueError("The system size must be divisible by the patch size")
        self.patch_states = jnp.array(list(product(range(self.LHilDim),repeat=self.patch_size)))
        self.LocalHilDim = self.LHilDim ** self.patch_size
        self.PL = self.L // self.patch_size
        index_array = self.LHilDim**(jnp.arange(self.patch_size)[::-1])
        self.index_map = jax.vmap(lambda s: index_array.dot(s))
        # select type of embedding
        if self.one_hot:
            self.embed = nn.Dense(self.embeddingDim, use_bias=False,name="Embedding", param_dtype=self.paramDType)
        else:
            self.embed = nn.Embed(self.LocalHilDim,
                               self.embeddingDim,
                               param_dtype=self.paramDType)
        # positional encoding
        self.p = self.variable(
            "params",
            "positional_embeddings",
            zeros, # jbr: this is the positional embedding, but it is all zeros
            (self.PL, self.embeddingDim),
            self.paramDType,
        ).value
        # sequential model
        self.sequential = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.hiddenDim, self.nHeads,self.init_variance,self.paramDType
                )
                for _ in range(self.depth)
            ]
        )
        # head 
        self.neck = nn.Dense(self.hiddenDim,name="Neck",
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)
        self.head = nn.Dense(self.LocalHilDim,name="Head", 
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.paramDType)

    def __call__(self, s: Array, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function and the complex phase.
        """

        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        # debug.print("{x}",x=s.shape)
        if self.one_hot:
            if returnLogAmp: s = self.index_map(s.reshape(self.PL,self.patch_size))
            y = jnp.pad(s[:-1],(1,0),mode='constant',constant_values=0)
            y = jax.nn.one_hot(y,self.LocalHilDim,axis=-1) 
            y = self.embed(y)
        else:
            if returnLogAmp: s = self.index_map(s.reshape(self.PL,self.patch_size))
            y = jnp.pad(s[:-1],(1,0),mode='constant',constant_values=0)
            y = self.embed(y)
        # jbr: adding the positional encoding
        y = y + self.p # jbr: this makes no sense, p is all zeoros
        y = self.sequential(y)
        # the neck precedes the head
        y = nn.gelu(self.neck(y))
        # the head tops the neck
        y = self.head(y)
        # return the log amplitude
        return (
            (take_along_axis(y, expand_dims(s, -1), axis=-1)
            .sum(axis=-2)
            .squeeze(-1))
        )