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

debug = True

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
    
    num_samples_measures = config["numSamples_measures"]
    
    steps = config["training_steps"]
    flag_NG = config["natural_gradient"]
    seed = config["seed"]
    training_steps = config["training_steps"]
    #"OutDir":"results", # dummy
    net_name = config["net"]
    net_para = config["net_parameter"]

    print(net_name)
    if net_name == "BHgpt":
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
    
    H = BoseHubbard_Hamiltonian1D(L,J,U,lDim,mu,V)
    iTerm = interactionTerm(L,lDim)
    jTerm = hoppingTerm(L,lDim)
    occ = occupations(L,lDim)
    measures_obs = {"jTerm": jTerm, "iTerm": iTerm, "energy": H,  "occupation": occ }

    net = gpt_particle.GPT(L,N,lDim,ebDim,dep,nH)
    sym = jVMC.util.symmetries.get_orbit_1D(L,"reflection","translation")
    symNet = jVMC.nets.sym_wrapper.SymNet(sym,net,avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep_real)

    psi = jVMC.vqs.NQS(symNet)

    sampler = jVMC.sampler.MCSampler(psi, (L,), jax.random.PRNGKey(seed), numSamples=num_samples)
    minSR_equation = jVMC.util.MinSR(sampler, makeReal='real',diagonalShift=diagShift,diagonalMulti=diagMult)
    
    resTraining = np.zeros((training_steps,2),dtype=float)
    #resMeasures_mean = np.zeros((training_steps,2+L),dtype=float)
    #resMeasures_var = np.zeros((training_steps,2+L),dtype=float)

    resMeasures = {}
    for key in measures_obs.keys():
        resMeasures[key] = {"mean" : [], "variance" : []}
        
    ipr_samp = np.zeros((training_steps,2),dtype=float)
    logipr_samp = np.zeros((training_steps,2),dtype=float)
    
    ##### training
    stepper = jVMC.util.stepper.Euler(timeStep=lr)  
    
    for n in range(training_steps):
        dpOld = psi.get_parameters()
    
        dp, _ = stepper.step(0, minSR_equation, dpOld, hamiltonian=H, psi=psi, numSamples=num_samples)
        psi.set_parameters(jnp.real(dp))
        mes = measure(measures_obs,psi=psi,sampler=sampler,numSamples=num_samples_measures)

        for key in measures_obs.keys():
            #print(key)
            #print(mes[key])
            if len(mes[key]["mean"]) == 1:
                resMeasures[key]["mean"] += mes[key]["mean"].tolist()
                resMeasures[key]["variance"] += mes[key]["variance"].tolist()
            else:
                resMeasures[key]["mean"] += [mes[key]["mean"].tolist()]
                resMeasures[key]["variance"] += [mes[key]["variance"].tolist()]
        resTraining[n] = [jnp.real(minSR_equation.ElocMean0) , minSR_equation.ElocVar0 ]
        
        #sampIPR = np.concatenate([np.exp(sampler.sample()[1].squeeze()) for _ in range(2)])
        sampIPR = jnp.exp(2*sampler.sample(numSamples=num_samples_measures)[1].squeeze()) # squared amplidutes -- probabilities |\psi|**2 (no phase as the ground state is positive)
        ipr_samp[n,0] = jnp.sum(sampIPR)/jnp.size(sampIPR) # MC sampling --> E[\psi**2] =corresponds= sum_{s \in Hilbert space} |\psi(s)|^4
        ipr_samp[n,1] = (jnp.sum(sampIPR**2)/jnp.size(sampIPR) - ipr_samp[n,0]**2)

        #entropy of probabilities: sum_{s \in Hilbert space} |\psi(s)|^2 log(|\psi(s)|^2 )
        logipr_samp[n,0] = jnp.sum(np.log(sampIPR))/jnp.size(sampIPR) 
        logipr_samp[n,1] = (jnp.sum(np.log(sampIPR)**2)/jnp.size(sampIPR) - logipr_samp[n,0]**2)

        # save measures, energies and to file 
        # compute Sx observables on the fly
        # save the results
        
    ##### saving results: measures
            
        if (((n%10) ==0) or (n == (training_steps-1))):        
            resMeasures["training_energy"]= {"mean": resTraining[:,0].tolist(), "variance": resTraining[:,1].tolist()}
            resMeasures["ipr"] ={"mean": ipr_samp[:,0].tolist(), "variance": ipr_samp[:,1].tolist()}
            resMeasures["logipr"] ={"mean": logipr_samp[:,0].tolist(), "variance": logipr_samp[:,1].tolist()}
            
            with open(config["OutDir"]+'/measures.json', 'w') as f:
                json.dump(resMeasures, f)
                
            h5save.save_model_params(psi.parameters,
                                    f"training_step_{n}",
                                    {})
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
                
    return "finished"
print(main())



