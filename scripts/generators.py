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

#Custom generator that takes a file with objects and labels
#Objects can have arbitrary numbers of pictures (in column picIDs without '.jpg')
#Batch size dependent on object number, not picture number

#Takes an image at filepath and label and returns a tuple:
#[np-array of size height * width * channels, label]
def get_image(filepath, y, height=256, width=256):
    image = cv2.imread(filepath)
    resized = cv2.resize(image, (height, width))
    return [resized, y]

#Takes np array representation of image and changes its brightness
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_img

#Takes np-array representation of image and label
#Returns k tuples: [augmented image, label]
#Augmented image is a random crop of size height * width * channels, plus random flip along vertical axis and brightness augmentation
def augment_image(image, y, k=1, height=224, width=224):
    if height > image.shape[0] or width > image.shape[1]:
        print('random_augment dimensions are greater than original dimensions')
    else:
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        aug_x = np.empty([0, height, width, 3], dtype=np.float32)
        aug_y = np.empty([0], dtype=np.int32)
        for i in range(k):
            left_height = random.randint(0, orig_height-height)
            left_width = random.randint(0, orig_width-width)
            subimage = image[left_height:left_height+height, left_width:left_width+width, :]
            do_flip = random.randint(0, 1)
            if do_flip == 1:
                subimage = random_brightness(cv2.flip(subimage, 1))
            else:
                subimage = random_brightness(subimage)

            aug_x = np.append(aug_x, [subimage], axis=0)
            aug_y = np.append(aug_y, [y])
            
    return [subimage, y]

#Takes np-like array of data and list of categorical variables in order
#Returns vectorized representation of data
def to_categorical(data, listorder):
    vectorized = np.zeros([len(data), len(listorder)])
    for i in range(len(data)):
        try:
            index = listorder.index(data[i])
            vectorized[i, index] += 1
        except:
            print('\nError in to_categorical: element in data not in listorder\n')
    return vectorized

                
#dataframe = pandas dataframe of all data, image_path = directory containing images, y_name = name of y-variable, y_classes = categories of images that will be returned by generator
#augment_number = number of times each images is augmented, images returned are dimension height * width * 3
#forTrain=True permutes and augments all images, =False does not
#Returns an image data generator for classification, permutes and augments all images by default
#Vectorized data indices are ordered by category appearance frequency
def categorical_generator(dataframe, image_path, y_name, y_classes=list(), augment_number=1, height=224, width=224, forTrain=True):
    
    #Generator
    while True:
        
        #Randomly permute images if forTrain=True
        if forTrain:
            indices_arr = np.random.permutation(dataframe.shape[0])
        #Non-permuted indices if forTrain=False
        else:
            indices_arr = np.arange(dataframe.shape[0])

        #Get examples
        for obj in range(len(indices_arr)):
            i = indices_arr[obj]

            #Skip image if not in y-classes
            if not dataframe.iloc[i].loc[y_name] in y_classes:
                continue
            
            for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                x_train = np.empty([0, height, width, 3], dtype=np.float32)
                y_train = np.empty([0], dtype=np.int32)
                
                try:               
                    #Get image, augment, and append to list of examples to yield if forTrain=True
                    if forTrain:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                dataframe.iloc[i].loc[y_name],
                                height+32,
                                width+32)
                        [aug_x, aug_y] = augment_image(image, y, augment_number, height, width)
                        
                        x_train = np.append(x_train, [aug_x], axis=0)
                        y_train = np.append(y_train, [aug_y])
                    
                    #Get image and append to list of examples to yield if forTrain=False
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
                    y_cat = to_categorical(y_train, y_classes)
                    yield(x_train, y_cat)
                
                #Error: failed to read pic
                except:
                    print('\nError: failed to read ' + pic+'.jpg\n')
                    continue

#dataframe = pandas dataframe of all data, image_path = directory containing images, y_name = name of y-variable
#augment_number = number of times each images is augmented, images returned are dimension height * width * 3
#forTrain=True permutes and augments all images, =False does not
#Returns an image data generator for regression, permutes and augments all images by default
#Images without valid data are ignored
def regression_generator(dataframe, image_path, y_name, augment_number=1, height=224, width=224, forTrain=True):
    
    #Generator
    while True:
        
        #Randomly permute images if forTrain=True
        if forTrain:
            indices_arr = np.random.permutation(dataframe.shape[0])
        #Non-permuted indices if forTrain=False
        else:
            indices_arr = np.arange(dataframe.shape[0])
            
        #Get examples
        for obj in range(len(indices_arr)):
            i = indices_arr[obj]
            
            #Skip image if not valid data
            if pd.isna(dataframe.iloc[i].loc[y_name]):
                continue
            
            for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                x_train = np.empty([0, height, width, 3], dtype=np.float32)
                y_train = np.empty([0], dtype=np.int32)
                
                try:
                    #Get image, augment, and append to list of examples to yield if forTrain=True
                    if forTrain:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                dataframe.iloc[i].loc[y_name],
                                height+32,
                                width+32)
                        [aug_x, aug_y] = augment_image(image, y, augment_number, height, width)
                        
                        x_train = np.append(x_train, [aug_x], axis=0)
                        y_train = np.append(y_train, [aug_y])
                    
                    #Get image and append to list of examples to yield if forTrain=False
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
                    yield(x_train, y_train)
                
                #Error: failed to read pic
                except:
                    print('\nError: failed to read ' + pic+'.jpg\n')
                    continue
                
#Takes same inputs as train_generator + batch_size
#generator = categorical, regression
#Returns a generator that yields batches of data from categorical_generator or regression_generator
def make_batch(generator, dataframe, image_path, y_name, y_classes=None, batch_size=32, augment_number=1, height=224, width=224, forTrain=True):
    if generator == 'categorical':
        gen = categorical_generator(dataframe, image_path, y_name, y_classes, augment_number, height, width, forTrain)
        while True:
            x_train = np.empty([0, height, width, 3], dtype=np.float32)
            y_train = np.empty([0, len(y_classes)], dtype=np.int32)
            for image in range(batch_size):
                [x_data, y_data] = next(gen)
                x_train = np.append(x_train, x_data, axis=0)
                y_train = np.append(y_train, y_data, axis=0)
            yield [x_train, y_train]
        
    elif generator == 'regression':
        gen = regression_generator(dataframe, image_path, y_name, augment_number, height, width, forTrain)
        while True:
            x_train = np.empty([0, height, width, 3], dtype=np.float32)
            y_train = np.empty([0], dtype=np.int32)
            for image in range(batch_size):
                [x_data, y_data] = next(gen)
                x_train = np.append(x_train, x_data, axis=0)
                y_train = np.append(y_train, y_data, axis=0)
            yield [x_train, y_train]
            
    else:
        print('\nError in make_batch: not a valid generator\n')
        
#dataframe = pandas dataframe of all data, image_path = directory containing images, y_name = name of y-variable, y_classes = categories of images that will be returned by generator
#y_classes = categories of images returned by test_generator, generator does not distinguish by y_name if y_classes=None
#include_NA determines whether or not generator yields images with invalid y_name data
#Returns an image data generator for testing, does not permute dataframe, augment images, or return y-label
def test_generator(dataframe, image_path, y_name=None, y_classes=None, include_NA=False, height=224, width=224):

    #Generator
    while True:
        
        #Get examples
        for i in range(len(dataframe)):
            
            #Skip image if not in y-classes
            if y_name != None and ((y_classes != None and not dataframe.iloc[i].loc[y_name] in y_classes) or (include_NA == False and pd.isna(dataframe.iloc[i].loc[y_name]))):
                continue
            
            for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                x_train = np.empty([0, height, width, 3], dtype=np.float32)
                
                try:
                    #Get image and append to list of examples to yield
                    [image, y] = get_image(
                            os.path.join(image_path, pic+'.jpg'),
                            dataframe.iloc[i].loc[y_name],
                            height,
                            width)
                    x_train = np.append(x_train, [image], axis=0)

                #Error: failed to read pic
                except:
                    print('\nError: failed to read ' + pic+'.jpg\n')
                    continue

                x_train = x_train / 255.0

                #Yield
                yield(x_train)
