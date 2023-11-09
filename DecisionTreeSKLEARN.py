#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:28:07 2023

@author: williamfloyd
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

def get_data():
    #Location of the movie data.  Will look at budget and rotten tomatoes score
    path = '/Users/williamfloyd/Documents/Data/Movie_Data_2023 - Sheet1.csv'
    
    MovieData = pd.read_csv(path)
    
    #Turns out we pulled in bunch of blank columns and I'd like to drop those
    names = MovieData.columns
    
    reduced_columns = [i for i in names if not i.__contains__('Unnamed')]
    
    #Also I don't care about the last two
    reduced_columns = reduced_columns[:-2]
    #Now from what I remember, we have to determine which variable 
    final_data = MovieData[reduced_columns].copy()
    final_data.drop(axis=0,index=67,inplace=True) #Junk row
    
    final_data = final_data.loc[final_data['Rotten Tomatoes  critics'].notnull()].copy()
    #print(final_data)
    
    return final_data



#Going to attempt to figure out the 'Domestic gross ($m)' column
MyData = get_data()
#For whatever reason when we do to numpy, we don't get a 2D array.  It simply gives us a
#1-D array.  the regressions looks like it wants a 2D input array, which in this case is
#a 1-D matrix confusingly enough.  Just has an explicit row and column specified now
X = np.reshape(MyData[['Budget ($m)','Rotten Tomatoes  critics']].to_numpy(),(-1,2))
Y = MyData['Domestic gross ($m)'].to_numpy()

regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_1.fit(X,Y)

#Want to actually see what we've got
attributes = regr_1.tree_.__dir__()

lefties = regr_1.tree_.children_left

# Setting dpi = 300 to make image clearer than default
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)


tree.plot_tree(regr_1,
           feature_names = ['Budget ($m)','Rotten Tomatoes  critics'], 
           filled = True);
