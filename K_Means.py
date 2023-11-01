#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:22:18 2023

@author: williamfloyd
Experiment in K-means clustering

"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import time


#Alright the first thing we need is a min and max for both x and y, and then 
#some random data



#fig,ax = plt.subplots()
#ax.scatter(X,Y)


#ax.plot(X1,Y1)

#plt.show()


#Now my understanding is the algorithm isn't that complicated in theory
"""
Step 0 is initialize with K random points (centroids)
Step 1 is to classify each point based on the closest centroid
Step 2 is then drawn new centroids based on on the 3 groups
Repeat 1 and 2 until happy

We'll need a function that can calculate a distance between two points (easy) (DONE)
A function that calculates the new centroid given a set of points (seems hard)
    MISTAKE: turns out with a little calc we can show this point is just the "mean"
    of our new points.  I.e. the mean of the x coordinates gives us the x coord
    of our centroid, etc (DONE)
A function to plot lines give centroids (easy, but tedious)

"""

def dist(point1,point2):
    #your garden variety distance
    x_dist = point1[0] - point2[0]
    y_dist = point1[1] - point2[1]
    return (x_dist**2+y_dist**2)**0.5


def get_centroid(points):
    n = len(points)
    vec_size = len(points[0])
    
    #this will contain our means
    ans = []

    for i in range(vec_size):
        run_total = 0
        
        for j in range(n):
            run_total += points[j][i] #ith coordinate of jth point
        
        ans.append(run_total/n)
    
    return ans



def get_line(point1,point2):
    #returns y = mx+b form of a line equidistant from two points
    x1,x2,y1,y2 = point1[0],point2[0],point1[1],point2[1]
    m = (x1-x2) / (y2-y1) #property of perpendicular line, negative recip of give line
    x0 = (x1+x2) / 2
    y0 = (y1+y2) / 2
    f = lambda x: m*(x-x0) + y0
    return f
    
 
def get_scatter_vec(points):
    X,Y = [],[]
    for point in points:
        X.append(point[0])
        Y.append(point[1])
    return X,Y

    
def get_closest_centroid(point,centroids):
    k = len(centroids)
    min_dist = pow(10,5)
    
    min_index = 2*k
    
    for i in range(k):
        cur_dist = dist(point,centroids[i])
        if cur_dist < min_dist:
            min_dist = cur_dist
            min_index = i
    
    return min_index #gives us which centroid is closest
    
#get_centroid([[0,0],[1,0],[0.5,(3**0.5)/2]])
#f = get_line([10,0],[5,5])
#X = [i for i in range(x_max+1)]
#Y = [f(x) for x in X]

def seed_centroids(k,x_min,x_max,y_min,y_max):
    
    ans = []
    #Honestly I'm just going to pick random points for now
    for i in range(k):
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        ans.append([x,y])
    #"random points"
    return ans


def get_data(my_seed):


    random.seed(my_seed)
    
    x_min,x_max = 0,10
    y_min,y_max = 0,10
    
    k = 4
    ghost_points = []
    #going to make very obvious clusters at first
    
    
    data = []
    for i in range(0,k):
        #Going to pick some random point and then generate stuff around it
        rand_x = random.random()*(x_max-x_min) + x_min
        rand_y = random.random()*(y_max-y_min) + y_min
        
        ghost_points.append([rand_x,rand_y])
        for j in range(10): #5 random points each
            x_dev = (x_max)*random.random() - (x_max/2)
            y_dev = (y_max)*random.random() - (y_max/2)
            data.append([rand_x+x_dev,rand_y+y_dev])
   
    return data



def get_variance(data,centroids):
    #given some centroids and data this gives the total variance
    #If the variance remains unchanged between generations we can stop
    var = 0
    keys = [i for i in data.keys()] #they are stored in dictionaries
    k = len(keys)
    
    for i in range(k):
        #for each group we have to calculate the variance
        #this should be vectorized but oh well
        for j in data[keys[i]]:
            #j is a point
            var += dist(j,centroids[i])
    
    
    return var





def k_means_function(k,data):
    #this does some shit with colors    
    colors = cm.rainbow(np.linspace(0, 1, k))
    print(colors)
    
    #I'm assuming that data is going to be a vector of 2-tuples
    X,Y = get_scatter_vec(data)
    x_min,x_max,y_min,y_max = min(X),max(X),min(Y),max(Y)
    data_x_max = x_max + 1
    data_y_max = y_max + 1


    #get the starting centroids
    centroids = seed_centroids(k,x_min,x_max,y_min,y_max)
    
    temp = {}
    temp['group_1'] = data
    for j in range(1,k):
        key = f"group_{j+1}"
        temp[key] = []
    
    
    cur_var = pow(10,10)
    tolerance = pow(10,-1)
    data = temp #dumb but oh well   
        
    for iteration in range(10):
        new_data = {}
        group_keys = []
        for i in range(k):
            key = f"group_{i+1}"
            new_data[key] = []
            group_keys.append(key)
        
        #take old data and classify each point 
        for my_key in data.keys():
            #each key is a collection of points
            cur_data = data[my_key]
            for point in cur_data: #loop through the points
                t = get_closest_centroid(point,centroids)
                new_data[group_keys[t]].append(point) #add point to new centroid
        
        
        #cent_X,cent_Y = get_scatter_vec(centroids)
        #ax.scatter(cent_X,cent_Y)
        new_centroids = []
        
        fig,ax = plt.subplots()
        ax.set_xlim([x_min-1,data_x_max])
        ax.set_ylim([y_min-1,data_y_max])
        ax.set_aspect('equal', adjustable='box')
    
        for i in range(k): #last time
            key = f"group_{i+1}"
            data_X,data_Y = get_scatter_vec(new_data[key])
            #plot the centroid and label it
            ax.scatter(centroids[i][0],centroids[i][1],color = colors[i],label = 'centroid')
            ax.annotate(f"centroid {i}",(centroids[i][0],centroids[i][1]))
            
            #plot the data
            ax.scatter(data_X,data_Y,color=colors[i])
        
            #we also need new centroids
            #print(f"current key is {key}")
            #print(new_data[key])
            try:
                #compute the new centroid
                new_centroids.append(get_centroid(new_data[key]))
            except:
                #Sometimes a centroid has no points close to it and we have to do this as a 
                #sort of base case
                new_centroids.append(centroids[i])
            
        
        centroids = new_centroids
        new_var = get_variance(new_data,centroids)
        
        plt.show()
        if cur_var - new_var < tolerance:
            #Then we have no change
            print(cur_var,new_var)
            print(f"CONVERGED AFTER {iteration} iterations")
            break
        cur_var = new_var
        
        
        
        time.sleep(1)
        


data = get_data(0.25)
k_means_function(3,data)