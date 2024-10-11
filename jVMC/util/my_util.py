import jVMC
import jax.numpy as jnp
import optax
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




def update_optax(psi,psiSampler,H,state_optimizer,optimizer,renormalization=None,mode=""):
    success = True
    psi_s, psi_logPsi, psi_p = psiSampler.sample()

    Eloc = H.get_O_loc(psi_s, psi, psi_logPsi)
    Eso = jVMC.util.SampledObs(Eloc, psi_p)
    Emean = Eso.mean()[0]
    Evar = Eso.var()[0]
    Opsi = psi.gradients(psi_s)
    grads = jVMC.util.SampledObs(Opsi, psi_p)
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
        return Emean.real,Evar,state_optimizer,n_p, n_grad, success

    update, state_optimizer = optimizer.update(
        checked_grad.reshape(psi_params.shape), state_optimizer, psi_params  # type: ignore
    )
    
    params = optax.apply_updates(psi_params, update)  # type: ignore
    
    psi.set_parameters(params)
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
