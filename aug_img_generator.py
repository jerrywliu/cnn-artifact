# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 02:15:55 2019

@author: weiho
"""

import cv2
import numpy as np
import pandas as pd
import random
import os
from keras.preprocessing.image import ImageDataGenerator

#Custom generator that takes a file with objects and labels
#Objects can have arbitrary numbers of pictures (in column picIDs without '.jpg')
#Batch size dependent on object number, not picture number

#Helper method for creating frequency list
def update_dictionary(dictionary, element):
    if element in dictionary.keys():
        dictionary[element] += 1
    else:
        dictionary[element] = 1

#Takes an image at filepath and label and returns a tuple:
#[np array of size height x width x channels, label]
def get_image(filepath, y, height=256, width=256):
    image = cv2.imread(filepath)
    resized = cv2.resize(image, (height, width))
    return [resized, y]

#Takes np array representation of image and label
#Returns 10 augmented images of size height x width x channels and label
#Cropped images from four corners and center, flipped along vertical axis
def augment_image(image, y, height=224, width=224):
    if height > image.shape[0] or width > image.shape[1]:
        print('augment_image dimensions are greater than original dimensions')
    else:
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        aug_x = np.empty([0, height, width, 3], dtype=np.float32)
        aug_y = np.empty([0], dtype=np.int32)
        centers = [
                [np.int32(np.floor(height/2)), np.int32(np.floor(width/2))],
                [orig_height - np.int32(np.ceil(height/2)), np.int32(np.floor(width/2))],
                [orig_height - np.int32(np.ceil(height/2)), orig_width - np.int32(np.ceil(width/2))],
                [np.int32(np.floor(height/2)), orig_width - np.int32(np.ceil(width/2))],
                [np.int32(np.floor(orig_height/2)), np.int32(np.floor(orig_width/2))]]
        for i in range(len(centers)):
            center_h = centers[i][0]
            center_w = centers[i][1]
            subimage = image[center_h-np.int32(np.floor(height/2)):center_h+np.int32(np.ceil(height/2)),
                             center_w-np.int32(np.floor(width/2)):center_w+np.int32(np.ceil(width/2)),
                             :]
            subimage1 = random_brightness(subimage)
            subimage2 = random_brightness(cv2.flip(subimage, 1))
            
            aug_x = np.append(aug_x, [subimage1, subimage2], axis=0)
            aug_y = np.append(aug_y, [y, y])
            
        return [aug_x, aug_y]

#Returns a random augmented image in the format of augment_image
def random_augment(image, y, height=224, width=224):
    if height > image.shape[0] or width > image.shape[1]:
        print('random_augment dimensions are greater than original dimensions')
    else:
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        left_height = random.randint(0, orig_height-height)
        left_width = random.randint(0, orig_width-width)
        subimage = image[left_height:left_height+height, left_width:left_width+width, :]
        do_flip = random.randint(0, 1)
        if do_flip == 1:
            subimage = random_brightness(cv2.flip(subimage, 1))
        else:
            subimage = random_brightness(subimage)

    return [subimage, y]

#Takes np array representation of image and changes its brightness
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_img

#Takes np-like array of data and list of categorical variables in order
#Returns vectorized representation of data
def to_categorical(data, listorder):
    vectorized = np.zeros([len(data), len(listorder)])
    for i in range(len(data)):
        try:
            index = listorder.index(data[i])
            vectorized[i, index] += 1
        except:
            print('Error in to_categorical: element in data not in listorder')
    return vectorized

#Takes a pandas dataframe of all data, the name of the y variable, directory containing images
#Returns a generator, batches are based on number of objects (not number of images)
#Vectorized data indices are ordered by category appearance frequency
#If yieldY=False, generator will yield image data for all examples in dataframe, including ones labeled 'NA'; otherwise, examples labeled 'NA' are ignored
def generator(dataframe, image_path, y_name, y_classes=set(), height=224, width=224, shuffle=True, augment=False, yieldY=True):
    
    #Initialize default y_classes
    if not y_classes:
        y_classes = set(dataframe.loc[:, y_name])
        if np.nan in y_classes:
            y_classes.remove(np.nan)
    
    #Get frequency list
    freqlist = get_freq_list(dataframe, y_name, y_classes)

    #Generator
    while True:
        #Shuffled or unshuffled indices
        if shuffle:
            indices_arr = np.random.permutation(dataframe.shape[0])
        else:
            indices_arr = np.arange(dataframe.shape[0])
            
        #Get examples
        for obj in range(len(indices_arr)):
            i = indices_arr[obj]
            #Skips object if yieldY is true and example is not in y_classes
            if yieldY:
                if not dataframe.iloc[i].loc[y_name] in y_classes:
                    continue
            for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                x_train = np.empty([0, height, width, 3], dtype=np.float32)
                y_train = np.empty([0], dtype=np.int32)
                try:
                    #Get image, augment if needed and append to list of examples to yield
                    if augment:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                dataframe.iloc[i].loc[y_name],
                                height+32,
                                width+32)
                        [aug_x, aug_y] = augment_image(image, y, height, width)
                        x_train = np.append(x_train, aug_x, axis=0)
                        y_train = np.append(y_train, aug_y)
                    else:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                dataframe.iloc[i].loc[y_name],
                                height,
                                width)
                        x_train = np.append(x_train, [image], axis=0)
                        y_train = np.append(y_train, [y])
                        
                    x_train = x_train / 255.0

                    #Yield
                    if yieldY:
                        y_cat = to_categorical(y_train, freqlist)
                        yield(x_train, y_cat)
                    else:
                        yield(x_train)
                
                #Error: failed to read pic
                except:
                    print('\nError: failed to read ' + pic+'.jpg\n')
                    continue

#Takes same inputs as above generator
#Returns a generator where each images is augmented randomly exactly once
#Vectorized data indices are ordered by category appearance frequency
#If yieldY=False, generator will yield image data for all examples in dataframe, including ones labeled 'NA'; otherwise, examples labeled 'NA' are ignored
def random_generator(dataframe, image_path, y_name, y_classes=set(), height=224, width=224, shuffle=True, augment=False, yieldY=True):
    
    #Initialize default y_classes
    if not y_classes:
        y_classes = set(dataframe.loc[:, y_name])
        if np.nan in y_classes:
            y_classes.remove(np.nan)
    
    #Get frequency list
    freqlist = get_freq_list(dataframe, y_name, y_classes)

    #Generator
    while True:
        #Shuffled or unshuffled indices
        if shuffle:
            indices_arr = np.random.permutation(dataframe.shape[0])
        else:
            indices_arr = np.arange(dataframe.shape[0])

        #Get examples
        for obj in range(len(indices_arr)):
            i = indices_arr[obj]
            #Skips object if yieldY is true and example is not in y_classes
            if yieldY:
                if not dataframe.iloc[i].loc[y_name] in y_classes:
                    continue
            for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                x_train = np.empty([0, height, width, 3], dtype=np.float32)
                y_train = np.empty([0], dtype=np.int32)
                try:
                    #Get image, augment if needed and append to list of examples to yield
                    if augment:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                dataframe.iloc[i].loc[y_name],
                                height+32,
                                width+32)
                        [aug_x, aug_y] = random_augment(image, y, height, width)

                        x_train = np.append(x_train, [aug_x], axis=0)
                        y_train = np.append(y_train, aug_y)
                    else:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                dataframe.iloc[i].loc[y_name],
                                height,
                                width)
                        x_train = np.append(x_train, [image], axis=0)
                        y_train = np.append(y_train, [y])
                        
                    x_train = x_train / 255.0

                    #Yield
                    if yieldY:
                        y_cat = to_categorical(y_train, freqlist)
                        yield(x_train, y_cat)
                    else:
                        yield(x_train)
                
                #Error: failed to read pic
                except:
                    print('\nError: failed to read ' + pic+'.jpg\n')
                    continue

#Takes same inputs as above generators + batch_size
#Returns a generator that yields batches of data from random_generator
def make_batch(dataframe, image_path, y_name, y_classes=set(), batch_size=32, augment_batch=False, height=224, width=224, shuffle=True, augment=False, yieldY=True):
    if augment_batch:
        gen = random_generator(dataframe, image_path, y_name, y_classes, height, width, shuffle, augment, yieldY)
    else:
        gen = generator(dataframe, image_path, y_name, y_classes, height, width, shuffle, augment, yieldY)
    while True:
        x_train = np.empty([0, height, width, 3], dtype=np.float32)
        y_train = np.empty([0, len(y_classes)], dtype=np.int32)
        for image in range(batch_size):
            [x_data, y_data] = next(gen)
            x_train = np.append(x_train, x_data, axis=0)
            y_train = np.append(y_train, y_data, axis=0)
        yield [x_train, y_train]

#Takes a dataframe of objects and y-variable and returns the number of images with valid y-variable data
def numImgs(dataframe, y_name, y_classes=set()):
    if not y_classes:
        y_classes = set(dataframe.loc[:, y_name])
        if np.nan in y_classes:
            y_classes.remove(np.nan)
    
    count = 0
    for i in range(len(dataframe)):
        if dataframe.iloc[i].loc[y_name] in y_classes:
            pics = dataframe.iloc[i].loc['picIDs'].split(',')
            count += len(pics)
    return count

#True with probability prop and False with probability 1-prop
def coin_flip(prop):
    r = random.random()
    if r <= prop:
        return True 
    else:
        return False

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

#Get frequency list, ordered by frequency of occurrence
def get_freq_list(dataframe, y_name, y_classes):
    freqdict = get_freq_dict(dataframe, y_name, y_classes)
    freqlist = list()
    while freqdict:
        for key in freqdict.keys():
            try:
                if freqdict[key] > freqdict[maxi]:
                    maxi = key
            except:
                maxi = key
        freqlist.append(maxi)
        freqdict.pop(maxi)
    return freqlist

#Returns a list of indices corresponding to argmax of l
def sort_list_by_index(l):
    index_set = set()
    sortedlist = list()
    for i in range(len(l)):
        index_set.add(i)
    while bool(index_set):
        maxi = ''
        for i in index_set:
            try:
                if l[i] > l[maxi]:
                    maxi = i
            except:
                maxi = i
        sortedlist.append(maxi)
        index_set.remove(maxi)
    return sortedlist
