#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:19:50 2023

@author: williamfloyd

Alright we're going to make some phoney balogna data here.  The goal 
is to get something approximating banking data.  

Note: It looks like the convention when storing data in a matrix is 
for each column to be a different variable and each row a different person
"""

import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


'''
Recall this is a way to get k colors.  I think the cm thing, among other things, 
let's you map values from 0 to 1 to different colors
colors = cm.rainbow(np.linspace(0, 1, k))
print(colors)
'''

def generate_covariates(field_name,num_covariates,**kwargs):
    #Because I'm basic We'll just start with normally distributed variables
    temp = []
    
    #mean,sd
    if kwargs['distribution_type'] == 'Normal':
        #Never though I'd make a lambda function with no variables but here we are
        f = lambda : random.normalvariate(kwargs['mean'], kwargs['sd'])
    
    
    if kwargs['distribution_type'] == 'Uniform':
        #Never though I'd make a lambda function with no variables but here we are
        f = lambda : random.uniform(kwargs['left'], kwargs['right'])
    
    
    for i in range(num_covariates):
        temp.append(f()) #Get the observation
        
    ans = pd.DataFrame(temp)
    ans.columns = [field_name]
    
    return ans



def get_data(seed,n):
    #n is the number of points we're generating
    
    #For this made up example I want to use DTI, Income, FICO, house price, Down payment
    
    #Always seems like a good idea to have this
    random.seed(seed)
    
    #Probably easiest just to keep the data point as an integer
    start = [i for i in range(n)]
    final_data = pd.DataFrame(start,columns=['Borrower ID'])
    
    #I'd like FICO scores
    FICO = generate_covariates("FICO", n, distribution_type = 'Normal',mean = 680,sd=40)
    
    #Let's do Debt to income as well
    DTI = generate_covariates("DTI", n, distribution_type = 'Uniform',left = 0,right=0.5)
    
    
    #Let's get downpayment
    DP = generate_covariates("Down Payment %", n, distribution_type = 'Normal',mean = 0.15,sd=0.05)
    f = lambda x: max(x,0) #Don't want less than 0 percent down payments
    DP['Down Payment %'] = DP['Down Payment %'].map(f)
    
    #Price of the house.  Note this shouldn't
    #be a normal variable, gotta do something else.  
    Price = generate_covariates("Price", n, distribution_type = 'Uniform',left = 100000,right=600000)
    #Price = generate_covariates("Price", n, distribution_type = 'Normal',mean = 350000,sd=50000)
    
    sd_func = lambda x: -1*abs(x-350000)*0.2 + 50000
    f = lambda x: (x/4) + random.normalvariate(0, 1*abs(x-350000)*0.25)
    #f = lambda x: x/4 + random.normalvariate(0, 10)
    
    Income = Price["Price"].map(f).copy().rename('Income')
    
    
    final_data = pd.concat([final_data,FICO,DP,Price,Income,DTI],axis=1)
    #final_data = pd.concat([final_data,Price,Income],axis=1)
    
    
    
    #print(final_data)
    return final_data
    
    
def normalize_matrix(A,type_normal):
    #Each column will get normalized
    #Gonna try two ways
    rows,cols = A.shape
    ans = np.ones((rows,1))
    
    for i in range(cols): #gotta iterate over them anyway
        if type_normal == 'standard_normal':
            #First we have to get the mean of the column
            cur_col = A[:,i] #Pick off our current column
            sd = cur_col.std()
            my_avg = cur_col.mean()
            shifted= np.ones((rows,1))*my_avg
            new_col = (cur_col-shifted)*(1/sd)
            #print(new_col)
            
            ans = np.concatenate((ans,new_col),axis=1)
            
        elif type_normal == 'linear':
            #Now this is what came to  my mind first.  The normalizing
            #Process with means and sds fundamentally changes the data imo
            pass
            
            
    #print(ans[:,1:])
    return ans[:,1:] #gotta drop that first column
            
            

def normalize_pandas(data):
    #I think normalizing is just going to be a good idea
    means,sds = data.mean(axis=0),data.std(axis=0) #goes down columns
    
    normalize = (data-means).copy() #Start the final answer
    sds_inv = sds.map(lambda x: 1/x).copy() #We need the inverse of the sd's
    
    #Crazily enough, pandas is pretty smart with dimensions and whatnot
    #Doing this gives us element wise multiplication for each row
    normalize = (normalize*sds_inv).copy()
    
    
    return normalize
    



t = 2*np.ones((4,2))
t[0,0] += 7
t[1,1] += 3

t = np.matrix(t)

normalize_matrix(t, 'standard_normal')

#
Housing_Data = get_data(77,200)
Housing_Data = Housing_Data.drop(['Borrower ID'],axis=1) #This column stinks

means,sds = Housing_Data.mean(axis=0),Housing_Data.std(axis=0) #goes down columns

#It's helpful to be able to translate between axis number and variable
axes = {}
axis = 0
for i in Housing_Data.columns:
    axes[f"axis {axis}"] = i
    axis += 1


House_Normal = normalize_pandas(Housing_Data)
var_covar = House_Normal.cov()

'''
This was all to manually calculate the variance covariance and principal components
No need for it now, but I'm happy I did it

#Experiment to get variance covariance matrix
A = Housing_Data.drop(['Borrower ID'],axis = 1).copy().to_numpy()
B = np.matrix(A)
C = normalize_matrix(B,'standard_normal')

#The algorithm mentioned that A needs to be normalized.  I thought normalizing
#Would "scrunch up the data" but that's wrong lol.  No clue why I thought that.  



#Rows is number of borrowers
rows,cols = C.shape

ONES = np.ones((rows,rows))
D = C - ONES*C*(1/rows)

#Note that by default pandas variance/covariance matrix used n-1 df for the 
#calculation, and I was using n.  If we use n-1 we match exactly
var_covar = (1/(rows-1))*D.T*D

#This was a fun experiement, but now I can do it all in pandas pretty easily.  
'''

#Column order is FICO, DP, Price, Income, DTI


#Note we have to put the arrays inside brackets to work here.  Not what I would have guessed,
#But what can you do
#plt.scatter(x=House_Normal['DTI'],y=House_Normal['FICO'],color='red')
#plt.scatter(x=House_Normal['Down Payment %'],y=House_Normal['Income'],color='green')

#Interesting note: I did not standardize my variables so the eigenvalues are
#all out of wack.  Looks like the eigenvectors are just the canonical basis, 
#whcih actually kind of makes sense.  The 3 variables are completely independent
#so there's no "direction of highest variance" besides right along the data axes
#I'm assuming the eigenvectors are in the same order as the eigenvalues
eigenvalues,eigenvectors = np.linalg.eig(var_covar)
#print(eigenvectors)

#plt.scatter(Housing_Data['Income'],Housing_Data['Price'])
#plt.quiver(50000,200000,24000,97000,angles = 'xy',scale_units = 'x',scale=1)

#Now let's cast our data onto the new axes and see what we get
Normal_Matrix = np.matrix(House_Normal.to_numpy())
eigenvec_matrix = np.matrix(eigenvectors)

eigenvec_basis = pd.DataFrame(eigenvec_matrix)

#Now what this does is give us the coefficients for each of the vectors in our 
#New Basis.  I'm curious to see what some of these look like
Recast_Data = pd.DataFrame(Normal_Matrix*eigenvec_matrix)
rows,cols = Recast_Data.shape
new_names = []
vec_names = []

for i in range(cols):
    new_names.append(f"axis {i}")
    vec_names.append(f"eigenvector {i}")


Recast_Data.columns = [new_names]
eigenvec_basis.columns = [vec_names]


#Alright what I want to do now may not make total sense, but we'll see
#I'd like to be able to pick two data dimensions and two eigenvectors
#And show the eigenvectors (or their projections at least) displayed
#Over the 2-d cut of that data
def plot_2d_cuts(data,eigenvectors,axis_dict,data_var0,data_var1,eigenvec0,eigenvec1):
    x_string = f"axis {data_var0}"
    y_string = f"axis {data_var1}"
    
    x = data[axis_dict[x_string]]
    y = data[axis_dict[y_string]]
    
    
    means,sds = data.mean(axis=0),data.std(axis=0) #goes down columns
    mean0,mean1 = means[axis_dict[x_string]],means[axis_dict[y_string]]
    sd0,sd1 = sds[axis_dict[x_string]],sds[axis_dict[y_string]]
    
    
    
    eigens = eigenvectors.iloc[[data_var0,data_var1]][[f"eigenvector {eigenvec0}",f"eigenvector {eigenvec1}"]]
    print(eigens)
    
    eigens = eigens.mul([sd0,sd1],axis=0) #this tells it to multiply first row by sd0,second by sd1
    #print(eigens)
    
    #eigens = eigens.add([mean0,mean1],axis=0)
    #print(eigens)
    #Now remember, these were scaled eigenvectors, we need to scale them back to the
    #Original data to make sense
    
    print(eigens)
    
    fig,ax = plt.subplots()
    
    ax.scatter(x=x,y=y)
    ax.quiver(mean0,mean1,eigens.iloc[0][f"eigenvector {eigenvec0}"]
               ,eigens.iloc[1][f"eigenvector {eigenvec0}"],angles = 'xy',scale_units = 'xy',scale=1,color='yellow')
    ax.quiver(mean0,mean1,eigens.iloc[0][f"eigenvector {eigenvec1}"]
               ,eigens.iloc[1][f"eigenvector {eigenvec1}"],angles = 'xy',scale_units = 'xy',scale=1)
    
    ax.set_xlabel(f"{axis_dict[x_string]}")
    ax.set_ylabel(f"{axis_dict[y_string]}")
    plt.show()

first_var = 0
second_var = 1

eig1 = 2
eig2 = 3

plot_2d_cuts(Housing_Data,eigenvec_basis,axes,first_var,second_var,eig1,eig2)
#plot_2d_cuts(Housing_Data,eigenvec_basis,axes,2,3,1,4)

plt.scatter(x=House_Normal[axes[f'axis {first_var}']],y=House_Normal[axes[f'axis {second_var}']],color='red')
plt.scatter(x=Recast_Data[f'axis {eig1}'],y=Recast_Data[f'axis {eig2}'])

#plt.scatter(x=Recast_Data['axis 1'],y=Recast_Data['axis 4'])