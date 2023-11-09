#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:33:06 2023

@author: williamfloyd
Going to attempt to code the algorithm to produce a simple decision tree.  Eventually
I'd like to build this up to a decision forecast algorithm, and then who knows
after that.  
"""

import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import tree

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


def get_square_error(data,split_field,split_num,output_field):
    #This is assuming the data has already been split
    #From what I gather we could use some other function, but we'll just use
    #the mean() for now
    data['split_bool'] = data[split_field].apply(lambda x: True if x < split_num else False)
    mean1 = data[output_field].loc[data['split_bool']==True].mean()
    mean2 = data[output_field].loc[data['split_bool']==False].mean()
    
    #Add residuals
    data['average'] = data[split_field].apply(lambda x: mean1 if x < split_num else mean2)
    
    #get residuals SQ
    square_resid=(data[output_field]-data['average']).pow(2).sum()
    
    return square_resid
    
    


def segment_on_field(data,field,output_field,min_group_size):
    #This will take a numeric field and determine the best splitting point
    
    #Also note that if our group data is less than 2 times the minimum group 
    #size then we can't split it because one of the resulting groups would be too small
    if len(data) < 2*min_group_size:
        print(data,min_group_size)
        return [False,0,0]
    
    
    cur_data = data[[field,output_field]].copy()
    cur_data = cur_data[[field,output_field]].sort_values(by=[field]).copy() #We only need these two fields
    cur_data.reset_index(inplace=True)
    cur_data = cur_data.drop(['index'],axis=1).copy()
    #Alright now the general idea is we loop through all the intermediate
    #values and calculate the squared error.  Whichever one gives the lowest 
    #squared error is our winner.  Remember too we don't want one that's too
    #small
    min_error = pow(10,20)
    rows,cols = cur_data.shape
    
    #We already know that we can't bother with the first several that will result in a minimum
    #group size that's too small.  Same on the other size.  
    cut = 0
    
    #Leaving the indexes with +1-1 for conceptual reasons.  The bound is
    #rows-group_size, but because of pythons indexing we add 1, but because of 
    #algorithm we subtract 1 lol
    for i in range(min_group_size-1,rows-min_group_size+1-1):
        num1,num2 = cur_data[field][i],cur_data[field][i+1]
        if num1 == num2:
            #No clue how to handle this
            continue
        split = (num1+num2)/2
        my_resid = get_square_error(cur_data,field,split,output_field)
        if my_resid < min_error:
            #Record the residual
            min_error = my_resid
            
            #Save the split
            cut = split
    
    #Alright now I think this will give us the value and minimum residual split
    return [True,min_error,cut]


def split_data(data,fields_of_interest,output_field,depth,min_group_size):
    
    
    if len(data) < 2*min_group_size:
        data[f'level {depth} split'] = 'No Split'
        
        return False,data #indicates no split
     
    #Alright the general Idea here will be to feed it some subset and it will
    #spit out the data with an appended column with the segmentation
    #we'll loop through the columns and whichever one's optimal split gives 
    #us the smallest squared errors is the winner
    
    error = pow(10,100)
    key_field = ''
    key_cut = 0
    my_bool = False
    
    for field in fields_of_interest:
        my_bool,square_residual,value_cut = segment_on_field(data,field,output_field,min_group_size)
        print(field,value_cut,square_residual)
        if my_bool == False: #This is in case our dataset was too small and we can't split it
            continue
        
        if square_residual < error: #New champ baby
            error = square_residual
            key_field = field
            key_cut = value_cut
    
    #print(key_field,key_cut)
    true_string = f"{key_field} > {key_cut}"
    false_string = f"{key_field} <= {key_cut}"
    data[f'level {depth} split'] = data[key_field].apply(lambda x: true_string if x > key_cut else false_string).copy()
    
    return True,data



#Next step here is to put in a...recursion function? I'd like to essentially
#Tell it to split the data, and then look at the left over populations 
#and just do the same thing

def get_full_tree(data,variables,y_var,cur_depth,max_depth,min_group_size):
    
    #Think of cur depth as which depth we're searching for
    
    if cur_depth > max_depth:
        #we've gone too far lol
        return data
    
    #print("trying split",len(data))
    #Alright now we take the current dataset we have and split it
    recurse_bool,temp = split_data(data,important_vars,y_var,cur_depth,min_group_size)
    
    
    #temp now has a new column.  That being said if our recursion bool 
    #is false then we didn't split.  no recurusive step
    
    if recurse_bool:
        #We now need to hit both of the new datasets with our recursion
        indices = temp[f'level {cur_depth} split'].unique().tolist() #gives us the distinct values
        #print(indices)
        
        left = temp.loc[temp[f'level {cur_depth} split']==indices[0]].copy()
        right = temp.loc[temp[f'level {cur_depth} split']==indices[1]].copy()
        
        nu_left = get_full_tree(left,important_vars,y_var,cur_depth+1,max_depth,min_group_size)
        nu_right = get_full_tree(right,important_vars,y_var,cur_depth+1,max_depth,min_group_size)
        
        temp = pd.concat([nu_left,nu_right]).copy()
        
    return temp #I think this should do it



#Going to attempt to figure out the 'Domestic gross ($m)' column
MyData = get_data()



#I really only care about a handful of these variables
#important_vars = ['Rotten Tomatoes  critics','Rotten Tomatoes Audience ','Budget ($m)','IMDB Rating']

important_vars = ['Rotten Tomatoes  critics','Budget ($m)']

#Alright now this does give us our full tree
MyData = get_full_tree(MyData,important_vars,'Domestic gross ($m)',1,3,2)




temp1 = MyData.loc[MyData['level 1 split']=='Budget ($m) <= 95.0'].copy()
temp2 = MyData.loc[MyData['level 1 split']=='Budget ($m) > 95.0'].copy()

fig,ax = plt.subplots()

ax.scatter(x=temp1['Budget ($m)'],y=temp1['Domestic gross ($m)'],color='red',label='little budget')
ax.scatter(x=temp2['Budget ($m)'],y=temp2['Domestic gross ($m)'],color='blue',label='big budget')
    

ax.set_xlabel('Budget')
ax.set_ylabel('Gross')
ax.legend()

plt.show()