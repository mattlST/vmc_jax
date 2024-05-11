import sys
import os

import numpy as np
import json

import json

# import os

import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jrnd
from jax import jit
import h5py as h5

import jVMC
from jVMC import nets
from jVMC import operator as op
from jVMC import sampler
from jVMC.vqs import NQS
from jVMC.util import h5SaveParams, NaturalGradient

############################################################
def create_folder(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # If not, create it
        os.makedirs(folder_path)


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



def main():
    ############################
    # loading configs
    with open(sys.argv[1], "r", encoding="utf8") as f:
        config = json.load(f)
    # create folder path
    create_folder(config["OutDir"])
    # save the config file to working directory
    json.dump(config,open(config["OutDir"]+"/config.json",'w'))
    L = config["L"] # number of spins
    N = config["N"] # number of iterations
    lDim = config["lDim"]
    J = config["J"]
    U = config["U"]
    V = config["V"]
    mu = config["mu"]
    lr = config["lr"]
    diagShift = config["diagS"]
    diagMult = config["diagM"]
    num_samples = config["numSamples"]
    steps = config["training_steps"]
    flag_NG = config["natural_gradient"]
    
    #"OutDir":"results", # dummy
    net = config["net"]
    net_para = config["net_parameter"]
    if net == "gpt_BH":
        ebDim = net_para[0]
        dep = net_para[1] 
        nH = net_para[2]
        net = gpt_particle.GPT(L,N,lDim,ebDim,dep,nH,)

    rseed = config["seed"]

    h5save = h5SaveParams(config["OutDir"]+"/params.h5",'w')

    #########
    # measures:
    # energy
    # ipr
    # occupations
    # interaction 
    #########
    
    H = BoseHubbard_Hamiltonian1D(L,J,U,lDim,mu,V):
    iTerm = interactionTerm(L,lDim)
    jTerm = hoppingTerm(L,lDim)
    occ = occupations(L,lDim)
    measures_obs = {"jTerm": jTerm, "iTerm": iTerm,  "occupation": occ }


    sym = jVMC.util.symmetries.get_orbit_1D(L,"reflection","translation")
    symNet = jVMC.nets.sym_wrapper.SymNet(sym,net,avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep_real)

    psi = jVMC.vqs.NQS(symNet)

    sampler = jVMC.sampler.MCSampler(psi, (L,), jax.random.PRNGKey(seed), numSamples=num_samples)
    minSR_equation = jVMC.util.MinSR(sampler, makeReal='real',diagonalShift=diagShift,diagonalMulti=diagMult)
    
    resTraining = np.zeros((n_steps,3),dtype=float)
    resMeasures = []
    ipr_samp = np.zeros(n_steps,dtype=float)
    ##### training
    stepper = jVMC.util.stepper.Euler(timeStep=lr)  
    
    for n in range(n_steps):
        dpOld = psi.get_parameters()
    
        dp, _ = stepper.step(0, minSR_equation, dpOld, hamiltonian=H, psi=psi, numSamples=numSamp)
        psi.set_parameters(jnp.real(dp))
    
        resMeasures += [measure(measures_obs,psi=psi,sampler=sampler)]
        resTraining[n] = [n, jnp.real(minSR_equation.ElocMean0) , minSR_equation.ElocVar0 ]
        sampIPR = np.concatenate([np.exp(sampler.sample()[1].squeeze()) for _ in range(2)])
        ipr_samp[n] = [np.sum(sampIPR**2)/len(sampIPR)

    # save measures, energies and to file 
        np.savetxt(config["OutDir"]+"/ipr_training.txt",ipr_samp)
        np.savetxt(config["OutDir"]+"/energy_training.txt",resTraining)
        # compute Sx observables on the fly
        SxOp = op.BranchFreeOperator()
        SxOp.add((op.Sx(L//2),))
        obs = jVMC.util.measure({"Sx": SxOp}, psi, psiSampler,numSamples=2**16)
        psi_obs = obs["Sx"]["mean"][0]
        psi_var = np.sqrt(obs["Sx"]["variance"][0]/2**(16))
        # save the results
        h5save.save_model_params(psi.parameters,
                                f"training_step_{n}"
                                )
                                #{"infidelity": infidelities[i],
                                #"time":T,
                                #"psi_obs":psi_obs,
                                #"psi_var":psi_var,
                                #"chi_obs":chi_obs,
                                #"chi_var":chi_var,
                                #"lr":learning_rate,
                                #"dshift":d_shift,
                                #"seed":seed,
                                #"numSamples":numSamples,
                                #"net_size":psi_params.shape[0],
                                #"net":config["net"]})

##### saving results: measures








