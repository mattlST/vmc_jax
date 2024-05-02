###############################################################
# Copy of the Natural Gradient as implemented by Schmitt & Reh
# Jonas Rigo Forschungs zentrum Jülich
# 04/09/2024
###############################################################


import jax
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs
from jVMC.stats import SampledObs

import warnings
from functools import partial

import jVMC.mpi_wrapper as mpi

class NaturalGradient:
    """
    Implementation of Natural Gradient following the procedure by Schmitt in https://doi.org/10.1103/PhysRevLett.125.100503
    The implementation can use a correction to the estimated Force term F by computing its Signal to noise ratio.
    Initializer arguments:
    * ``sampler``: A sampler object.
    * ``snrTol``: Regularization parameter :math:`\epsilon_{SNR}`, see above.
    * ``pinvTol``: Regularization parameter :math:`\epsilon_{SVD}` (see above) is chosen such that :math:`||S\\dot\\theta-F|| / ||F||<pinvTol`.
    * ``pinvCutoff``: Lower bound for the regularization parameter :math:`\epsilon_{SVD}`, see above.
    * ``diagonalShift``: Regularization parameter :math:`\\rho` for ground state search, see above.
    * ``diagonalizeOnDevice``: Choose whether to diagonalize :math:`S` on GPU or CPU.
    """

    def __init__(self, sampler, snrTol=4., pinvTol=1e-14, pinvCutoff=1e-8, diagonalShift=0., diagonalScale=0., diagonalizeOnDevice=True):
        
        self.sampler = sampler
        self.snrTol = snrTol
        self.pinvTol = pinvTol
        self.pinvCutoff = pinvCutoff
        self.diagonalShift = diagonalShift
        self.diagonalScale = diagonalScale
        self.diagonalizeOnDevice = diagonalizeOnDevice

        self.jitted_inv = jax.jit(jnp.linalg.inv)
        self.jitted_dot = jax.jit(jnp.dot)

    def _transform_to_eigenbasis(self, S, F):
        
        if self.diagonalizeOnDevice:
            try:
                self.ev, self.V = jnp.linalg.eigh(S)
            except:
                self.diagonalizeOnDevice = False
        if not self.diagonalizeOnDevice:
            tmpS = np.array(S)
            tmpEv, tmpV = np.linalg.eigh(tmpS)
            self.ev = jnp.array(tmpEv)
            self.V = jnp.array(tmpV)

        self.VtF = jnp.dot(jnp.transpose(jnp.conj(self.V)), F)


    def NatGrad(self, F, psi_grad, **kwargs):
        """
        Compute the Natural gradient update as described by 
        Schmitt in https://doi.org/10.1103/PhysRevLett.125.100503
        Arguments:
            * ``F``: gradient of the  Loss function known as force term
            * ``WFgrad``: jVMC.stats.SampledObs logarithmic derivative of the wave function

        Further optional keyword arguments:
            * ``ObsLoc``: jVMC.stats.SampledObs object for the local Loss function

        Returns:
            * Natural gadient update
        """
        # Get TDVP equation from MC data
        S = psi_grad.covar().real 
        # applying the diagonal shift
        S = S + jnp.diag(self.diagonalScale * jnp.diag(S) + self.diagonalShift * jnp.ones_like(S[0]))

        # Transform TDVP equation to eigenbasis and compute SNR
        self._transform_to_eigenbasis(S, F) 

        # invert the eigenvalues and
        # discard eigenvalues below numerical precision
        self.invEv = jnp.where(jnp.abs(self.ev / (self.ev[-1])) > 1e-14, 1. / self.ev, 0.)

        if "ObsLoc" in kwargs:
            EO = psi_grad.covar_data(kwargs["ObsLoc"]).transform(linearFun = jnp.transpose(jnp.conj(self.V)))
            self.rhoVar = EO.var().ravel()

            # computing the signal to noise ratio of the F vector
            self.snr = jnp.sqrt(jnp.abs(mpi.globNumSamples * (jnp.conj(self.VtF) * self.VtF).squeeze(-1) / self.rhoVar)).ravel()

            residual = 1.0
            regularizer = 1.0
            pinvEv = self.invEv * regularizer
            cutoff = 1e-2
            F_norm = jnp.linalg.norm(F)
            while residual > self.pinvTol and cutoff > self.pinvCutoff:
                cutoff *= 0.8
                # Set regularizer for singular value cutoff
                regularizer = 1. / (1. + (max(cutoff, self.pinvCutoff) / jnp.abs(self.ev / self.ev[-1]))**6)

                if "ObsLoc" in kwargs and not isinstance(self.sampler, jVMC.sampler.ExactSampler):
                    # Construct a soft cutoff based the SNR
                    regularizer = 1. / (1. + (self.snrTol / self.snr)**6)

                pinvEv = self.invEv * regularizer

                residual = jnp.linalg.norm((pinvEv * self.ev - jnp.ones_like(pinvEv)) * self.VtF) / F_norm

            self.invEv = pinvEv

        return jnp.dot(self.V, (self.invEv  * self.VtF.reshape(self.invEv .shape)))