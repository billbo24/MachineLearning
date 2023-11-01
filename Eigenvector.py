#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:12:33 2023

@author: williamfloyd
Want to write a script to calculate eigenvectors

I'll skip the proof since its a bit involved, but it looks like
there's a fairly simple iterative algorithm, namely:
    
    For a matrix A pick b0 as our first guess
    b(k+1) = A*bk/||A*bk||
    
    And that's it! Eventually it converges to a vector that I believe points in the 
    correct direction.  
    
    
Also quick TIL
numpy has a matrix object in addition to the array object.  A matrix object is a subclass
of the array class but with some syntactic sugar.  Apparently A.T gives the transpose
and A.I gives the inverse.  Also A*B gives matrix multiplication.  Arrays can matrix multiply too
using the @ symbol


"""


import numpy as np
import matplotlib.pyplot as plt
import random

A = np.matrix([[11,12],[3,4]])

b0 = np.matrix([[random.random()],[random.random()]])
b0 /= np.linalg.norm(b0)
A*b0
bn = b0
for i in range(10):
    save = bn
    temp = A*bn
    mag = np.linalg.norm(temp)
    bn = temp/mag #normalize
    
    
    
    plt.quiver(0,0,bn[0,0], bn[1,0])
    plt.show()
    
    if np.linalg.norm(bn-save) < 0.01:
        print("eigenvector baby!!!")
        #renormalize so x = 1
        const = 1/bn[0,0]
        
        bn *= const
        
        print(bn,A*bn)
        
        lambda1 = np.linalg.norm(A*bn)/np.linalg.norm(bn)
        print(f"Eigenvalue is {lambda1}")
        break




