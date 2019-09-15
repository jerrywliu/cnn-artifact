# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:59:13 2019

@author: weiho
"""

import pandas as pd

#Creates new dataframe treating objects in original dataframe containing multiple images as separate objects, each with a single image
def split_objects(dataframe, outFile):
    split_df = list()
    for i in range(len(dataframe)):
        df_object = dict()
        for j in range(len(dataframe.iloc[i].loc['picIDs'].split(','))):
            for column in dataframe.columns:
                if column == 'picIDs' or column == 'picURLs':
                    df_object[column] = dataframe.iloc[i].loc[column].split(',')[j]
                else:
                    df_object[column] = dataframe.iloc[i].loc[column]
        split_df.append(df_object)
    
    returndf = pd.DataFrame(split_df, columns=dataframe.columns)
    returndf.to_csv(outFile, columns=returndf.columns, na_rep='NA', encoding='utf-8', index=None)

#Helper method for creating frequency list
def update_dictionary(dictionary, element):
    if element in dictionary.keys():
        dictionary[element] += 1
    else:
        dictionary[element] = 1
        
#Takes a dataframe of objects and returns the number of images with y-variable data in y_classes
def numImgs(dataframe, y_name, y_classes):
    count = 0
    for i in range(len(dataframe)):
        if dataframe.iloc[i].loc[y_name] in y_classes:
            pics = dataframe.iloc[i].loc['picIDs'].split(',')
            count += len(pics)
    return count

#Takes a dataframe of objects and returns the number of images in dataframe
#include_NA=True: counts all images in dataframe
#include_NA=False: counts all images in dataframe with non-NA y_name data
def allImgs(dataframe, y_name=None, include_NA=False):
    count = 0
    if include_NA:
        for i in range(len(dataframe)):
            pics = dataframe.iloc[i].loc['picIDs'].spilt(',')
            count += len(pics)
    else:
        for i in range(len(dataframe)):
            if not pd.isna(dataframe.iloc[i].loc[y_name]):
                pics = dataframe.iloc[i].loc['picIDs'].split(',')
                count += len(pics)
    return count

#Get frequency dictionary
def get_freq_dict(dataframe, y_name, y_classes):
    freqdict = dict()
    for i in range(len(dataframe)):
        if dataframe.iloc[i].loc[y_name] in y_classes:
            update_dictionary(freqdict, dataframe.iloc[i].loc[y_name])
    for y_class in y_classes:
        if y_class not in freqdict.keys():
            freqdict[y_class] = 0
    return freqdict
