import json

from jax.config import config  # type: ignore
import jax.numpy as jnp

config.update("jax_enable_x64", True)

import jVMC
from jVMC.operator.bosons import *
from jVMC.util.symmetries import LatticeSymmetry
from jVMC.global_defs import tReal
from jVMC.util import measure
import gpt_particle





import network
######## load config
with open('test_file.json') as f:
    d = json.load(f)
    print(d)

L,N,lDim
ebDim, dep, nH
#########
# measures:
# energy
# ipr
# occupations
# interaction 
#########
measures = {"H": Hamiltonian, "I": Interaction, "IPR": ipr, "occupation": occ }


sym = jVMC.util.symmetries.get_orbit_1D(L,"reflection","translation")
net = gpt_particle.GPT(L,N,lDim,ebDim,dep,nH,)
symNet = jVMC.nets.sym_wrapper.SymNet(sym,net,avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep_real)

psi = jVMC.vqs.NQS(symNet, seed=282)
#psi.set_generator(False)

sampler = jVMC.sampler.MCSampler(psi, (L,), jax.random.PRNGKey(1), 
tdvpEquation = jVMC.util.MinSR(sampler, makeReal='real',diagonalShift=diagonalShift,diagonalMulti=diagonalMulti)
res = np.zeros((n_steps,3),dtype=float)

##### training

for n in range(n_steps):
    dpOld = psi.get_parameters()

    stepper,numSamp = timeStepper(n)
    dp, _ = stepper.step(0, tdvpEquation, dpOld, hamiltonian=hamiltonian1D, psi=psi, numSamples=numSamp)
    psi.set_parameters(jnp.real(dp))

    res = measure({"H": hamiltonian1D},psi=psi,sampler=sampler)
    res[n] = [n, jnp.real(tdvpEquation.ElocMean0) , tdvpEquation.ElocVar0 ]
    


##### saving results: measures








