# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:09:31 2019

@author: weiho
"""

import numpy as np
import pandas as pd
from aux_func import update_dictionary
        
#Takes in a file of objects inFile
#Returns a train and a val file to outFile+'_train.txt' and outFile+'val.txt' in 1-val_prop : val_prop split
#If numCat == 0, splits all objects in inFile with valid numerical data
#Otherwise, sorts all categories in categoryName by frequency of appearance, looks at the first numCat most frequent classes, and creates balanced train and val files with number of objects equal to the number of objects in the least frequent class present
def train_val_split(inFile, outFile, categoryName, numCat, val_prop=0.3):
    df = pd.read_csv(inFile)
    
    if numCat == 0:
        catindices = list()
        for i in range(len(df)):
            try:
                catvalue = int(df.iloc[i].loc[categoryName])
                catindices.append(i)
            except:
                continue

        perm = np.random.permutation(len(catindices))
        indexorder = list(map(lambda x: catindices[x], perm))

        traindf = pd.DataFrame(columns=df.columns)
        valdf = pd.DataFrame(columns=df.columns)
        
        traindf = traindf.append(df.iloc[indexorder[:int(len(catindices)*(1-val_prop))],:])
        valdf = valdf.append(df.iloc[indexorder[int(len(catindices)*(1-val_prop)):len(catindices)],:])

        traindf.to_csv(outFile+'_train.txt', columns=traindf.columns, na_rep='NA', encoding='utf-8', index=None)
        valdf.to_csv(outFile+'_val.txt', columns=traindf.columns, na_rep='NA', encoding='utf-8', index=None)
    
    else:
        categories = dict()
        for i in range(len(df)):
            update_dictionary(categories, df.iloc[i].loc[categoryName])
        if np.nan in categories.keys():
            categories.pop(np.nan)

        catlist = list()
        while categories:
            for key in categories.keys():
                try:
                    if categories[maxkey] < categories[key]:
                        maxkey = key
                except:
                    maxkey = key
            catlist.append(maxkey)
            categories.pop(maxkey)
            
        catindices = list()
        numCat = min(len(catlist), numCat)
        
        for i in range(numCat):
            catindices.append(list())
            
        for i in range(len(df)):
            if df.iloc[i].loc[categoryName] in catlist[:numCat]:
                catindices[catlist.index(df.iloc[i].loc[categoryName])].append(i)
                
        numImgs = len(catindices[numCat-1])
        print('Extracting ' + str(numImgs) + ' images from ' + str(numCat) + ' ' + categoryName + ' classses.')
        
        traindf = pd.DataFrame(columns=df.columns)
        valdf = pd.DataFrame(columns=df.columns)
        
        for i in range(numCat):
            perm = np.random.permutation(len(catindices[i]))
            indexorder = list(map(lambda x: catindices[i][x], perm))
            traindf = traindf.append(df.iloc[indexorder[:int(numImgs*(1-val_prop))],:])
            valdf = valdf.append(df.iloc[indexorder[int(numImgs*(1-val_prop)):numImgs],:])
             
        traindf.to_csv(outFile+'_train.txt', columns=traindf.columns, na_rep='NA', encoding='utf-8', index=None)
        valdf.to_csv(outFile+'_val.txt', columns=traindf.columns, na_rep='NA', encoding='utf-8', index=None)
        
        return catlist[:numCat]
