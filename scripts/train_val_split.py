# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:09:31 2019

@author: weiho
"""

import numpy as np
import pandas as pd
from aux_func.py import update_dictionary
        
#
def train_val_split(inFile, outFile, categoryName, numCat, val_prop=0.3):
    df = pd.read_csv(inFile)
    
    categories = dict()
    for i in range(len(df)):
        update_dictionary(categories, df.iloc[i].loc[categoryName])
        
    catlist = list()
    while categories:
        for key in categories.keys:
            try:
                if categories[maxkey] < categories[key]:
                    maxkey = key
            except:
                maxkey = key
        catlist.append(maxkey)
        categories.pop(maxkey)
        
    catindices = list()
    numCat = max(len(catlist), numCat)
    
    for i in range(numCat):
        catindices.append(list())
        
    for i in range(len(df)):
        if df.iloc[i].loc[categoryName] in catlist[:numCat]:
            catindices[catlist.index(df.iloc[i].loc[categoryName])].append(i)
            
    numImgs = len(catindices[categoryName-1])
    print('Extracting ' + numImgs + ' images from ' + numCat + ' ' + categoryName + ' classses.')
    
    traindf = pd.DataFrame(columns=df.columns)
    valdf = pd.DataFrame(columns=df.columns)
    
    for i in range(numCat):
        perm = np.random.permutation(len(catindices[i]))
        traindf = traindf.append(df.iloc[catindices[perm[:(int) (numImgs*(1-val_prop))]]])
        valdf = valdf.append(df.iloc[catindices[perm[(int) (numImgs*(1-val_prop)):numImgs]]])
        
    traindf.to_csv(path=outFile+'_train.txt', columns=traindf.columns, na_rep='NA', encoding='utf-8', index=None)
    valdf.to_csv(path=outFile+'_val.txt', columns=traindf.columns, na_rep='NA', encoding='utf-8', index=None)
    
    return catlist[:numCat]