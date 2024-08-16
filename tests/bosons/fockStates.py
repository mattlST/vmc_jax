import numpy as np
from math import comb

# code from Jona Naegerl
class fockSpace:
    def __fockgenerator (self,n,l,L,focklist,fockSpaces):
        if l > 1:
            for m in range(0,n+1):
                focklist += (m,)
                focklist = self.__fockgenerator(n-m,l-1,L,focklist,fockSpaces)
                if m != n and l != L:    
                    focklist += focklist[-L:-l] 
        else:
            focklist += (n,)
            fockSpaces += focklist[-L:]
        return focklist
    def __wrapperFockGenerator(self,N,L):
        return ([tuple((*x,)) for x in np.array(self.__fockgenerator(N,L,L,(),())).reshape(-1,L)])
        return (self.__fockgenerator(N,L,L,(),()))
    

    def __init__(self,N,L):
        self.N = N
        self.L= L
        self.dim = comb(N+L-1,L-1)
        self.fockSpace = self.__wrapperFockGenerator(N,L)
        self.fockArray = np.array(self.fockSpace,dtype=int)
        self.dict = dict(zip(self.fockSpace,np.arange(self.dim)))
        
    def __call__(self):
        pass
    def __str__(self):
        return print(f"particle number: {self.N} \n site number: {self.L}\n dimension: {self.dim} ")
