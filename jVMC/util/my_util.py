import jVMC
import jax.numpy as jnp
import optax
import numpy as np
def check_gradient(grad):
    flag = not jnp.all(jnp.isfinite(grad))
    gradR = jnp.where(jnp.isfinite(grad),grad,0)
    return gradR, flag

def re_grad(grad,renormalization,mode=""):
    if renormalization is not None:
        re_norm = renormalization
        norm_grad = jnp.linalg.norm(grad)
        if mode == "minmax":
            assert len(re_norm) ==2, "mode minmax needs a 2-tupel"
            if norm_grad < re_norm[0]:
                return re_norm[0] *grad/norm_grad
            if norm_grad > re_norm[1]:
                return re_norm[1] *grad/norm_grad
            else:
                return grad
        else:
            assert isinstance(re_norm,float), "mode needs a float number (except minmax mode)"
            if mode == "max":
                if norm_grad< re_norm:
                    return grad
            elif mode == "min":
                if norm_grad> re_norm:
                    return grad
        return re_norm *grad/norm_grad
    return grad




def update_optax(psi,psiSampler,H,state_optimizer,optimizer,tempAnnealing=0.,renormalization=None,mode="",flag_retEnt=False):
    success = True
    psi_s, psi_logPsi, psi_p = psiSampler.sample()

    Eloc = H.get_O_loc(psi_s, psi, psi_logPsi)
    Eso = jVMC.util.SampledObs(Eloc, psi_p)
    Emean = Eso.mean()[0]
    Evar = Eso.var()[0]
    Opsi = psi.gradients(psi_s)
    grads = jVMC.util.SampledObs(Opsi, psi_p)

    #psi_grads = jVMC.util.SampledObs(psi.get_grad)
    if tempAnnealing>1e-12:
        Entropy = jVMC.util.SampledObs((-2.) * psi_logPsi.real, psi_p)
        Ent_mean = Entropy.mean()[0]
        Ent_var = Entropy.var()[0]
        
        Ent_grad = - 2.*grads.covar(Entropy) - 2.*jnp.expand_dims(grads.mean(),axis=-1) 
        Egrad = (1-np.min([1.,tempAnnealing]))* 2.*grads.covar(Eso)+ tempAnnealing * Ent_grad
    else:
        Egrad = 2.*grads.covar(Eso)
    grad = jnp.real(Egrad)
    grad = jnp.nan_to_num(grad,0.)
    n_grad = jnp.linalg.norm(grad)
    grad = re_grad(grad,renormalization,mode)
    checked_grad, flag = check_gradient(grad)
    
    psi_params = psi.get_parameters()
    n_p = jnp.linalg.norm(psi_params)
    if flag:
        success = False
        print("checking grad failed:")
        print("checked grad",checked_grad)
        print("grad",grad)
        print("no update")
    else:    
        update, state_optimizer = optimizer.update(
            checked_grad.reshape(psi_params.shape), state_optimizer, psi_params  # type: ignore
        )
        
        params = optax.apply_updates(psi_params, update)  # type: ignore
        
        psi.set_parameters(params)
    if flag_retEnt:
        return Emean.real,Evar,state_optimizer,n_p, n_grad,success,Ent_mean,Ent_var
    return Emean.real,Evar,state_optimizer,n_p, n_grad,success

def sr_update(psi,lr_SR,equations,H,renormalization=None,mode=""):
    success = True
    dpOld = psi.get_parameters()            
    n_p = jnp.linalg.norm(dpOld)
    dp = equations(dpOld,0,hamiltonian=H, psi=psi,intStep=0)
    dp = jnp.nan_to_num(dp,0.)
    n_grad = jnp.linalg.norm(dp)

    dp = re_grad(dp,renormalization,mode)
    checked_grad, flag = check_gradient(dp)
    if flag:
        success = False
        print("checking grad failed:")
        print("checked grad",checked_grad)
        print("grad",checked_grad)
        return jnp.real(equations.ElocMean0) , equations.ElocVar0, n_p, n_grad,success

    #dp, _ = stepperSR.step(0, equations, dpOld, hamiltonian=H, psi=psi)
    psi.set_parameters(dpOld + lr_SR * jnp.real(checked_grad))
    return  jnp.real(equations.ElocMean0) , equations.ElocVar0, n_p, n_grad,success


def sampling_histogram(sampler,ldim,numSamp=2**10,repeats=10):
    #plt.imshow(np.array([np.histogram(x,ldim,range=(0,ldim-1),density=True)[0] for x in  sampler_GPT.sample(numSamples=1000)[0][0].T]))
    y = np.zeros((sampler.sampleShape[0],1+ldim),dtype=float)
    for i in range(repeats):
        s, log_psi, p = sampler.sample(numSamples=numSamp)
        y += np.array([np.histogram(x,ldim+1,range=(0,ldim),weights=p[0],density=True)[0] for x in  s[0].T])
    return y/repeats


def annealing_optax(psi,psiSampler,state_optimizer,optimizer,renormalization=None,mode=""):
    success = True
    psi_s, psi_logPsi, psi_p = psiSampler.sample()

    # Eloc = H.get_O_loc(psi_s, psi, psi_logPsi)
    # Eso = jVMC.util.SampledObs(Eloc, psi_p)
    # Emean = Eso.mean()[0]
    # Evar = Eso.var()[0]
    Opsi = psi.gradients(psi_s)
    grads = jVMC.util.SampledObs(Opsi, psi_p)

    #psi_grads = jVMC.util.SampledObs(psi.get_grad)
    Entropy = jVMC.util.SampledObs((-2.) * psi_logPsi.real, psi_p)
    Entmean = Entropy.mean()[0]
    Entvar = Entropy.var()[0]
    
    Ent_grad = - 2.*grads.covar(Entropy) - 2.*jnp.expand_dims(grads.mean(),axis=-1) 
    Egrad =  Ent_grad
    grad = jnp.real(Egrad)
    grad = jnp.nan_to_num(grad,0.)
    n_grad = jnp.linalg.norm(grad)
    grad = re_grad(grad,renormalization,mode)
    checked_grad, flag = check_gradient(grad)
    
    psi_params = psi.get_parameters()
    n_p = jnp.linalg.norm(psi_params)
    if flag:
        success = False
        print("checking grad failed:")
        print("checked grad",checked_grad)
        print("grad",grad)
        return Entmean.real,Entvar,state_optimizer,n_p, n_grad, success

    update, state_optimizer = optimizer.update(
        checked_grad.reshape(psi_params.shape), state_optimizer, psi_params  # type: ignore
    )
    
    params = optax.apply_updates(psi_params, update)  # type: ignore
    
    psi.set_parameters(params)
    return Entmean,Entvar,state_optimizer,n_p, n_grad,success
