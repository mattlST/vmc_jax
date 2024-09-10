import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse as sp 
import scipy.sparse.linalg as spl
from tqdm import tqdm
from itertools import product
import math

import fockStates


@np.vectorize
def dimBH(N,L):
    return math.comb(N+L-1,L-1)


############## usual fock basis ####################
def hopping(fock,L):
    dim = fock.dim
    i_index = []
    j_index = []
    val = []
    for x in fock.fockSpace:
        indexPairs = zip(np.arange(L),np.roll(np.arange(L),1))
        ix = fock.dict[x]

        for index1,index2 in indexPairs:
            ay = np.array(x)
            if ay[index1] ==0:
                continue
            ay[index1] -= 1
            ay[index2] += 1
            y = tuple(ay) 
            iy = fock.dict[y]
            i_index += [ix]
            j_index += [iy]
            val += [np.sqrt(x[index1]*y[index2])]
    i_index = np.array(i_index)
    j_index = np.array(j_index)
    val = np.array(val)

    hopping = sp.coo_matrix((val,(i_index,j_index)),shape=(dim,dim)).tocsr()
    hopping = hopping+hopping.T
    return hopping
def interaction(fock,L):
    dim = fock.dim
    interactionVec = np.sum(fock.fockArray**2-fock.fockArray,axis=1)
    return sp.diags(interactionVec,shape=(dim,dim))
def chemical(fock,L,mu):
    dim = fock.dim
    chemicalVec = np.sum(mu[None,:]*fock.fockArray,axis=1)
    return sp.diags(chemicalVec,shape=(dim,dim))


def BH(J,U,L,N,mu=None):
    """
    Bose Hubbard Hamiltonian
    """
    fock = fockStates.fockSpace(N,L)
    dim = fock.dim
    hoppingSparse = hopping(fock,L)
    interactionSparse = interaction(fock,L)
    if mu is not None:
        chemSparse = chemical(fock,L,mu)
        return -J*hoppingSparse + U/2 *interactionSparse + chemSparse
    return -J*hoppingSparse + U/2 *interactionSparse

def gs_BH(J,U,L,N,mu=None):
    H = BH(J,U,L,N,mu=None)
    energy,state = spl.eigsh(H,k=1,which ="SA")
    return energy,state
    
############## momentum fock basis ####################

def mHopping(fock,L):
    dim = fock.dim
    en = np.cos(2*np.pi/L*np.arange(L))*2
    dia = np.sum(fock.fockArray * en[None,:],axis=1)
    return sp.diags(dia,shape = (dim,dim))
def mInteraction(fock,L):
    dim = fock.dim
    i_index = []
    j_index = []
    val = []
    for x in fock.fockSpace:
        indexPairs = product(np.arange(L),np.arange(L),np.arange(L))
        ix = fock.dict[x]

        for indexP1,indexP2,indexM1 in indexPairs:
            indexM2 = (+indexP1+indexP2-indexM1)%L
            #print(indexM2)
            ay = np.array(x)
            if ((ay[indexM1] ==0) or (ay[indexM2] == 0)):
                continue
            if ((indexM1==indexM2) and (ay[indexM1]==1)):
                continue
            zw = ay[indexM1]
            ay[indexM1] -= 1
            zw *= ay[indexM2]
            ay[indexM2] -= 1
            
            zw *= (ay[indexP1]+1)
            ay[indexP1] += 1
            
            zw *= (ay[indexP2]+1)
            ay[indexP2] += 1
            
            y = tuple(ay) 
            iy = fock.dict[y]
            i_index += [ix]
            j_index += [iy]
            val += [np.sqrt(zw)]
    i_index = np.array(i_index)
    j_index = np.array(j_index)
    val = np.array(val)

    mInteraction = sp.coo_matrix((val,(i_index,j_index)),shape=(dim,dim)).tocsr()
    mInteraction = (1./L)*mInteraction #- sp.eye(dim)*np.sum(fock.fockArray[0])
    return mInteraction

############## momentum fock basis, translation-symmetry section ####################
def mHoppingBlock(fock,L,momentum):
    m0 = np.where((np.sum(fock.fockArray*np.arange(L)[None,:],axis=1)%L)==momentum)[0]
    dimB = len(m0)

    en = np.cos(2*np.pi/L*np.arange(L))*2
    dia = np.sum(fock.fockArray[m0] * en[None,:],axis=1)
    return sp.diags(dia,shape = (dimB,dimB))



def mInteractionBlock(fock,L,momentum):
    m0 = np.where((np.sum(fock.fockArray*np.arange(L)[None,:],axis=1)%L)==momentum)[0]
    dimB = len(m0)
    mdic = dict(zip(m0,range(len(m0))))
    i_index = []
    j_index = []
    val = []
    for ix,x in enumerate(fock.fockArray[m0]):
        indexPairs = product(np.arange(L),np.arange(L),np.arange(L))
        #ix = fock.dict[tuple(x)]

        for indexP1,indexP2,indexM1 in indexPairs:
            indexM2 = (+indexP1+indexP2-indexM1)%L
            #print(indexM2)
            ay = x*1.
            if ((ay[indexM1] ==0) or (ay[indexM2] == 0)):
                continue
            if ((indexM1==indexM2) and (ay[indexM1]==1)):
                continue
            zw = ay[indexM1]
            ay[indexM1] -= 1
            zw *= ay[indexM2]
            ay[indexM2] -= 1
            
            zw *= (ay[indexP1]+1)
            ay[indexP1] += 1
            
            zw *= (ay[indexP2]+1)
            ay[indexP2] += 1
            
            y = tuple(ay) 
            iy = fock.dict[y]
            iiy = mdic[iy]
            i_index += [ix]
            j_index += [iiy]
            val += [np.sqrt(zw)]
    i_index = np.array(i_index)
    j_index = np.array(j_index)
    val = np.array(val)

    mInteraction = sp.coo_matrix((val,(i_index,j_index)),shape=(dimB,dimB)).tocsr()
    mInteraction = (1./L)*mInteraction #- sp.eye(dim)*np.sum(fock.fockArray[0])
    return mInteraction

