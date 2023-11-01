#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:56:19 2023

@author: williamfloyd

Alright this is a mess, but we're gonna try and get all eigenvalues.  
The iterative method only finds the biggest...which can be useful I suppose 
but let's learn


TAKEAWAY
Computing eigenvectors is a much more complicated problem than I thought.  It looks
like for kinda simple matrices the QR method can get you eigenvalues without too
much fanfare, but getting eigenvectors is a different beast.  This is all to say
I think I have a better grasp on eigenvectors, although still not the best.  
The one benefit is for a matrix A, A*A.t is always symmetric and it turns out
this is exactly how a varaince covariance matrix is calculated
"""

import numpy as np
import matplotlib.pyplot as plt
import random

#This suppresses scientific notation in our matrices.  
np.set_printoptions(suppress=True)

#This is our matrix
a = np.matrix([[-2,-2,4]])
b = np.matrix([[-4,1,2]])
c = np.matrix([[2,2,5]])




#Pain in the ass making column vectors lol
a,b,c = a.T,b.T,c.T


#Looks like dot product requires one to be transposed and it returns a 1x1 matrix
#my_dot = np.dot(a.T, b)


def proj_a_along_b(a,b):
    #Expecting both to be column vectors
    #A little vector math will show us this is
    #b*(adotb)/(bdotb).  Turns out all the cosines
    #and shit cancel.  Kinda nice
    adotb = np.dot(a.T,b)
    bdotb = np.dot(b.T,b)
    
    mag = adotb[0,0]/bdotb[0,0]
    return mag*b

temp1 = np.matrix([[1],[1]])
temp2 = np.matrix([[1],[-1]])
temp3 = proj_a_along_b(temp1, temp2)


def normalize_vec(a):
    mag = np.linalg.norm(a)
    return (1/mag)*a

#This gets us an orthonormal basis
#np.concatenate will be a useful function.  Feed in all vectors as a tuple
#Then axis = 0 stacks vertically, axis = 1 horizontally
def Gram_Schmidt(my_matrix):
    #For good practice going to assume square, but arbitrary nxn
    v1 = my_matrix[:,0] #Definition
    rows,cols = my_matrix.shape #Returns row,col
    ans = normalize_vec(v1)
    for i in range(1,cols):
        #This gives us the nth column
        cur_col = my_matrix[:,i]
        temp_col = cur_col
        #We have to project our current vector onto all the previous ones
        for j in range(0,i):
            #projects current one onto previous one
            proj = proj_a_along_b(cur_col, ans[:,j])
            temp_col = temp_col - proj
        norm_vec = normalize_vec(temp_col)
        ans = np.concatenate((ans,norm_vec),axis=1)
    
    
    return ans

#With the gram schmidt stuff done now it's quite trivial to get this
def get_QR_decomp(A):
    Q = Gram_Schmidt(A)
    R = Q.T*A
    return Q,R


'''
According to wikipedia, we can find the eigenvalues of A as follows
let A0 = 0
Write Ak = QkRk (QR decomp)
A(k+1) = R(k+1)Q(k+1)
Admittedly I'm taking a little bit on faith here, but it looks like due to 
some linear algebra properties, our matrix A(k+1) has the same eigenvalues
as Ak (note this is particularly because of R and Q, namely 
       we can write Ak+1 = Qk.T*Ak*Qk)

'''
def get_eigenvalues(A,tolerance):
    #Note I don't fully understand the conditions under which this algorithm
    #Converges, but it didn't for the one I was doing previously.  
    #That was because the matrix had complex eigenvales .  
    Ak = A
    Qk,Rk = get_QR_decomp(Ak)
    
    for i in range(100):
        
        #This is kind of an iterative step, pushing each column of Qk 
        #closer to an eigenvector
        
        #Qk,Rk = get_QR_decomp(Ak)
        #Ak = Rk*Qk #Apparently it is this easy
        
        #Extra thing to consider: Remember we could iteratively hit any vector
        #with a matrix, renormalize it, and eventually we'd get to an eigenvector?
        #Well we can modify this algorithm just a bit to do that with all eigenvectors.  
        #Quick reason: The matrix kinda stretches your vector most in the direction
        #of the biggest eigenvector.  Cool.  But what if we then switched it to a vecto
        #that was perpendicular to it? I can squint a little bit and convince myself
        #that it may not move it in the direction of the biggest eigenvector, but 
        #then where does it go? Apparently next biggest.  
        W = A*Qk
        Qk,Rk = get_QR_decomp(W)
        
    #print(Ak)    
    print(Qk)

A = np.concatenate((a,b,c),axis = 1)

get_eigenvalues(A,100)
eigenvalues,eigenvectors = np.linalg.eig(A)
print(eigenvalues)


