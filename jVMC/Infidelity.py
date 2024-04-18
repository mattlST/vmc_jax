##############################################
# J Rigo
# rigojonas@gmail.com
# Regensburg 20/11/2023
# Implementation of https://arxiv.org/abs/2305.14294
# by Sinibaldi et al. 
##############################################

import jax
from jax import jit, vmap, grad#, partial
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
from jVMC.stats import SampledObs
import jVMC.operator as op


import functools
import itertools


class _InfidelityP(op.Operator):
    """This class provides functionality to compute ratios of neural quantum states :math:`\\frac{\\chi(s)}{\\psi(s)}`. More specifically it implements

        :math:`\sum_{s}\\delta_{ss'}\\frac{\\chi(s)}{\\psi(s')}` 


    Initializer arguments:
        None
    """   

    def compile(self):

        def map_function(s, *args,):

            sp = s.copy()
            configShape = sp.shape

            matEl = jnp.array([1.], dtype=global_defs.tCpx)

            return sp.reshape(1,*configShape), matEl

        return map_function

#########################################################

class Infidelity(op.Operator):
    """This class provides functionality to compute and optimize the infidelity given a target wave function :math:`\chi(s')` and a trial wave function :math:`\psi(s)`.

    The infidelity for two pure states is defined as
    
        :math:`\\mathcal{I} = 1 - \\sum_s \\Psi(s)\\frac{\\chi(s)}{\\psi(s)}\\times\\sum_{s'} \\Chi(s')\\frac{\\psi(s')}{\\chi(s')} = 1 - \\mathbb{E}_\\Psi\\frac{\\chi(s)}{\\psi(s)}\\times\\mathbb{E}_\\Chi\\frac{\\psi(s')}{\\chi(s')}`

    where we have introduced the short-hand for the Born distribution for the target and trial wave function
    
        :math:`\\Psi(s) = \\frac{\\psi(s)}{\\sum_s|\\psi(s)|^2} \\qquad \\Chi(s') = \\frac{\\chi(s')}{\\sum_{s'}|\\chi(s')|^2}` 

    With the local infidelity 
    
        :math:`F^\\psi_{\\rm loc}(s) = \\frac{\\chi(s)}{\\psi(s)}\\times\\sum_{s'} \\qquad F^\\chi_{\\rm loc}(s') = \\frac{\\psi(s')}{\\chi(s')}`

    the infidelity can be expressed as 

        :math:`\\mathcal{I} = 1 - \\mathbb{E}_\\Psi F^\\psi_{\\rm loc}(s)\\times\mathbb{E}_\\Chi F^\\chi_{\\rm loc}(s')`

    In this class the :math:`F^\\psi` and :math:`F^\\chi` are computed separately, such that :math:`F^\\chi` can be computed once and reused for any :math:`\\psi`.


    Initializer arguments:

        * ``chi``: Target wave function as neural quantum state.
        * ``chiSampler``: Sampler initialized with ``chi``
    """
    
    def __init__(self,chi,chiSampler,getCV=False,adaptCV=False):

        #########################################################

        # save the target wave funciton internally
        self.chi = chi
        # save the target sampler internally
        self.chiSampler = chiSampler
        # control variates constant
        self.CVc = -0.5
        # control variates flag
        self.getCV = getCV
        # control variates adaptation flag
        self.adaptCV = adaptCV
        # user error check
        if adaptCV and not getCV:
            print("WARNING: adaptCV is set to True, but getCV is set to False. Setting adaptCV to True.")
            self.getCV = True

        ######## get samples ########
        self.chi_s, self.chi_logChi, self.chi_p = self.chiSampler.sample()
        # get the Infidelity to global scope
        self._InfidelityP = _InfidelityP()
        # needs to be called for initalization
        _, _ = self._InfidelityP.get_s_primes(self.chi_s)
        # check if was called
        self._FP_loc_check = False

        super().__init__()  # Constructor of base class Operator has to be called!
        
    def _get_CVc(self):
        """ 
        We compute the control variates factor :math:`c` for the local infidelity :math:`F^\\chi_{\\rm loc}(s)`, using 
        the square fidelity :math:`|\mathcal{F}|^2`
        :math:` c = -\frac{\text{cov}\big[ \Re \mathcal{F},~|\mathcal{F}|^2 \big]}{\text{var}|\mathcal{F}|^2}`
        This reads written out
        :math:` 
        \text{cov}\big[ \Re \mathcal{F},~|\mathcal{F}|^2 \big] = \\
        \mathbb{E}_\psi 
        \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}} 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^2
        \mathbb{E}_\chi 
        \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}} 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^2
        -
        \mathbb{E}_\psi 
        \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}} 
        \mathbb{E}_\psi 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^2
        \mathbb{E}_\chi 
        \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}} 
        \mathbb{E}_\chi 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^2 \\
        \rm var |\mathcal{F}|^2
        =  
        \mathbb{E}_\psi 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^4
        \mathbb{E}_\chi 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^4
        -
        \Bigg(
        \mathbb{E}_\psi 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^2
        \mathbb{E}_\chi 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^2
        \Bigg)^2`
        """

        return None
    
    def get_FP_loc(self, psi):
        """ Compute :math:`F^\\chi` given the target wave function :math:`\\chi` and trial wave function :math:`\\psi`.

        Arguments:
            * ``psi``: Neural quantum state.

        Returns:
            * :math:`F^\\chi_{\\rm loc}(s)` for each configuration` :math:`s`.
            * :math:`F^\\chi`    
        """

        # confirm that the function was called
        self._FP_loc_check = True
        # get psi nqs amplitudes
        chi_logPsi = psi(self.chi_s)
        # compute F^\chi_loc
        self.chi_Floc = self._InfidelityP.get_O_loc(self.chi_logChi,chi_logPsi)
        # control variates stabilization of the local infidelity
        if self.getCV:
            self.chi_FlocCV = self._InfidelityP.get_O_loc(2.*self.chi_logChi.real,2.*chi_logPsi.real)
            self.Exp_chi_FlocCV = mpi.global_mean(self.chi_FlocCV,self.chi_p)
            if self.adaptCV:
                # square squared fidelity
                self.chi_F2locCV = self._InfidelityP.get_O_loc(4.*self.chi_logChi.real,4.*chi_logPsi.real)
                self.Exp_chi_F2locCV = mpi.global_mean(self.chi_F2locCV,self.chi_p)
                # square infedlity times infidelity
                self.chi_Floc2FlocCV = self._InfidelityP.get_O_loc(
                                                                    self.chi_logChi* 2. * self.chi_logChi.real,
                                                                    chi_logPsi * 2.*chi_logPsi.real
                                                                    )
                self.Exp_chi_Floc2FlocCV = mpi.global_mean(self.chi_Floc2FlocCV,self.chi_p)
                
        # compute F^chi
        self.Exp_chi_Floc = mpi.global_mean(self.chi_Floc,self.chi_p)
        return self.chi_Floc, self.Exp_chi_Floc
    
    
    def get_gradient(self, psi, psi_p,  corrections=False):
        """ Compute :math:`\\nabla\\mathcal{I}`

        The gradient is computed from almost exclusively internally stored quantities

            :mat:`2\\mathbb{E}_\\Psi\\big[ \\Re~ O_k(s)F^\\psi_{\\rm loc}(s)\\big] \\, F^\\chi-2\\mathbb{E}_\\Psi\\big[ \\Re~ O_k(\\sigma)\\big] \\, F^\\psi \\, F^\\chi`        

        where :math:`O_k(s)` is the log derivative of the trial wave function.
        Note that the control variate contribution is not included in the gradient.

        Arguments:
            * ``psi``: Neural quantum state.
            * ``psi_p``: Born distribution :math:`\Psi(s)` evaluated at the samples
            * ``corrections``: Bool to include gradient correction

        Returns:
            * :math:`F^\\chi_{\\rm loc}(s)` for each configuration :math:`s`.
            * :math:`O`    
        """

        # compute the gradient throught the covariance
        Opsi = psi.gradients(self._flatten_pmapd(self.sp))
        grads = SampledObs(Opsi, psi_p)
        Floc = SampledObs(self.psi_Floc, psi_p)
        grad = 2.*grads.covar(Floc)*self.Exp_chi_Floc

        # gradient with CV
        if False:#self.getCV:
            
            FlocCV = SampledObs(self.psi_FlocCV, psi_p)
            grad -= self.CVc * 2. * grads.covar(FlocCV)*self.Exp_chi_FlocCV

        return -1. * grad.real, grads

    def compile(self):
        """This function computes ratios of neural quantum states :math:`\\frac{\\chi(s)}{\\psi(s)}`. More specifically it implements

            :math:`\sum_{s}\\delta_{ss'}\\frac{\\chi(s)}{\\psi(s')}` 


        Initializer arguments:
            None
        """   
        
        # check if get_FP_loc was called
        if not self._FP_loc_check:
            raise ValueError("get_FP_loc was not called")

        def map_function(s, *args,):

            sp = s.copy()
            configShape = sp.shape

            matEl = jnp.array([1.], dtype=global_defs.tCpx)

            return sp.reshape(1,*configShape), matEl

        return map_function


    def get_O_loc(self, samples, psi, logPsiS=None, *args):
        """Compute :math:`F^\\psi_{\\rm loc}(s)`.

        If the instance parameter ElocBatchSize is larger than 0 :math:`F^\\psi_{\\rm loc}(s)` is computed in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math::math:`F^\\psi_{\\rm loc}(s)` for each configuration :math:`s`.
        """

        if logPsiS is None:
            _ , _ = self.get_s_primes(samples, *args)
            logPsiS = psi(samples)

        if self.ElocBatchSize > 0:
            return self.get_O_loc_batched(samples, psi, logPsiS, self.ElocBatchSize, *args)
        else:
            return self.get_O_loc_unbatched(logPsiS)


    def get_O_loc_unbatched(self, logPsiS,batched=False,logPsiSP=None):
        """Compute :math:`F^\\psi_{\\rm loc}(s)`.

        This member function assumes that ``get_s_primes(s)`` has been called before, as \
        internally stored matrix elements :math:`F^\\chi` are used.

        Computes :math:`F_{loc}(s)= F_\chi\\frac{\chi(s)}{\psi(s)}`, given the \
        logarithmic wave function amplitudes of the involved configurations :math:`\\ln(\psi(s))` \
        and :math:`\\ln\psi(s')`

        Arguments:
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`

        Returns:
            :math:`1 - F^\\psi_{\\rm loc}(s)F^\\chi` for each configuration :math:`s`.
        """
        # get the second part of the infideltiy
        self.psi_Floc = self._get_O_loc_pmapd(self.matEl, logPsiS, self.chi(self._flatten_pmapd(self.sp)))
        # control variates stabilization of the local infidelity
        if batched:
            return self.psi_Floc
        elif self.getCV:
            self.psi_FlocCV = self._get_O_loc_pmapd(self.matEl, 2.*logPsiS.real, 2.*self.chi(self._flatten_pmapd(self.sp)).real)
            return 1. - self.psi_Floc * self.Exp_chi_Floc - self.CVc * (self.psi_FlocCV*self.Exp_chi_FlocCV - 1.)
        else:    
            # return the infidelity
            return 1. - self.psi_Floc * self.Exp_chi_Floc
        
    def _get_O_loc_batched(self, samples, psi, logPsiS, batchSize, Observables, *args):
        """Compute :math:`F^\\psi_{\\rm loc}(s)` in batches.

        Computes :math:`F^\\psi_{\\rm loc}(s)=F^\\chi\\frac{\chi(s)}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``batchSize``: Batch size.
            * ``Observables``: functions modifying the Logarithmic amplitudes.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`1 - F^\\psi_{\\rm loc}(s)F^\\chi` for each configuration :math:`s`.
        """

        Oloc_set = [None for _ in Observables]

        numSamples = samples.shape[1]
        numBatches = numSamples // batchSize
        remainder = numSamples % batchSize

        # Minimize mismatch
        if remainder > 0:
            batchSize = numSamples // (numBatches+1)
            numBatches = numSamples // batchSize
            remainder = numSamples % batchSize

        for b in range(numBatches):

            batch = self._get_config_batch_pmapd(samples, b * batchSize, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, b * batchSize, batchSize)

            # internalize the new batch
            sp, _ = self.get_s_primes(batch, *args)

            # modified the get_O_loc_unbatched function
            OlocBatch_set = [self.get_O_loc_unbatched(ol(logPsiSbatch),batched=True) for ol in Observables]
            
            for ol_ind,ol in enumerate(Oloc_set):
                if ol is None:
                    if OlocBatch_set[ol_ind].dtype == global_defs.tCpx:
                        Oloc_set[ol] = self._alloc_Oloc_cpx_pmapd(samples)
                    else:
                        Oloc_set[ol] = self._alloc_Oloc_real_pmapd(samples)

            ##############
            # regular infideltiy
            Oloc_set = [self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize) for Oloc, OlocBatch in zip(Oloc_set, OlocBatch_set)]
        
        if remainder > 0:

            batch = self._get_config_batch_pmapd(samples, numBatches * batchSize, remainder)
            batch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(batch, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, numBatches * batchSize, numSamples % batchSize)
            logPsiSbatch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(logPsiSbatch, batchSize)

            # internalize the new batch
            sp, _ = self.get_s_primes(batch, *args)

            ##############
            # regular infideltiy
            # modified the get_O_loc_unbatched function
            OlocBatch_set = [self.get_O_loc_unbatched(ol(logPsiSbatch),batched=True) for ol in Observables]
        
            OlocBatch_set = [self._get_Oloc_slice_pmapd(OlocBatch, 0, remainder) for OlocBatch in OlocBatch_set]
            # get the second part of the infideltiy
            Oloc_set = [self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize) for Oloc, OlocBatch in zip(Oloc_set, OlocBatch_set)]
                
        # sretrn the local quantity
        return  Oloc_set

    def get_O_loc_batched(self, samples, psi, logPsiS, batchSize, *args):
        """Compute :math:`F^\\psi_{\\rm loc}(s)` in batches.

        Computes :math:`F^\\psi_{\\rm loc}(s)=F^\\chi\\frac{\chi(s)}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``batchSize``: Batch size.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`1 - F^\\psi_{\\rm loc}(s)F^\\chi` for each configuration :math:`s`.
        """

        if self.adaptCV:
            Oloc_set = self._get_O_loc_batched(samples,
                                        psi,
                                        logPsiS, 
                                        batchSize,
                                        [lambda x: x, lambda x: 2.*x.real, lambda x: 4.*x.real, lambda x: x*2.*x.real]
                                        )
            self.psi_Floc = Oloc_set[0]
            self.psi_FlocCV = Oloc_set[1]
            self.psi_F2locCV = Oloc_set[2]
            self.psi_Floc2FlocCV = Oloc_set[3]
            return None
        
        elif self.getCV:
            Oloc_set = self._get_O_loc_batched(samples,
                                        psi,
                                        logPsiS, 
                                        batchSize,
                                        [lambda x: x, lambda x: 2.*x.real]
                                        )
            self.psi_Floc = Oloc_set[0]
            self.psi_FlocCV = Oloc_set[1]
            return 1. - self.psi_Floc * self.Exp_chi_Floc - self.CVc * (self.psi_FlocCV*self.Exp_chi_FlocCV - 1.)
        
        else:
            Oloc_set = self._get_O_loc_batched(samples,
                                                    psi,
                                                    logPsiS, 
                                                    batchSize,
                                                    [lambda x: x])
            self.psi_Floc = Oloc_set[0]
            return 1. - self.psi_Floc * self.Exp_chi_Floc

# auxillary function
def expand_batch(batch, batchSize):
    outShape = list(batch.shape)
    outShape[0] = batchSize
    outp = jnp.zeros(tuple(outShape), dtype=batch.dtype)
    return outp.at[:batch.shape[0]].set(batch)
