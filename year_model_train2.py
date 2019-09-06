# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:16:45 2019
@author: weiho
"""
import os
'''
#Uses GPU 1
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
'''
import csv
import cv2
import keras
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from sklearn.metrics import classification_report, confusion_matrix
import sys
import tensorflow as tf

from inception_model import inception_v1

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

#Takes a pandas dataframe of all data, y-variable is 'year', directory containing images
#Returns a generator, batches are based on number of objects (not number of images)
#If yieldY=False, generator will yield image data for all examples in dataframe, including ones labeled 'NA'; otherwise, examples labeled 'NA' are ignored
def generator(dataframe, image_path, height=224, width=224, shuffle=True, augment=False, yieldY=True):

    y_name = 'year'
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
                if pd.isna(dataframe.iloc[i].loc[y_name]):
                    continue
            try:
                yearvalue = float(dataframe.iloc[i].loc[y_name])
                for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                    x_train = np.empty([0, height, width, 3], dtype=np.float32)
                    y_train = np.empty([0], dtype=np.int32)
                    
                    #Get image, augment if needed and append to list of examples to yield
                    if augment:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                yearvalue,
                                height+32,
                                width+32)

                        [aug_x, aug_y] = augment_image(image, y, height, width)
                        x_train = np.append(x_train, aug_x, axis=0)
                        y_train = np.append(y_train, [aug_y])
                    
                    else:
                        [image, y] = get_image(
                                os.path.join(image_path, pic+'.jpg'),
                                yearvalue,
                                height,
                                width)
                        
                        x_train = np.append(x_train, [image], axis=0)
                        y_train = np.append(y_train, [y])

                    x_train = x_train / 255.0

                    #Yield
                    if yieldY:
                        yield(x_train, y_train)
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
def random_generator(dataframe, image_path, height=224, width=224, shuffle=True, augment=False, yieldY=True):
    
    y_name = 'year'
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
                if pd.isna(dataframe.iloc[i].loc[y_name]):
                    continue
            try:
                yearvalue = float(dataframe.iloc[i].loc[y_name])
                for pic in dataframe.iloc[i].loc['picIDs'].split(','):
                    x_train = np.empty([0, height, width, 3], dtype=np.float32)
                    y_train = np.empty([0], dtype=np.int32)
                    
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
                        y_train = np.append(y_train, y)
                        
                    x_train = x_train / 255.0

                    #Yield
                    if yieldY:
                        yield(x_train, y_train)
                    else:
                        yield(x_train)
                
            #Error: failed to read pic
            except:
                print('\nError: failed to read ' + pic+'.jpg\n')
                continue

#Takes same inputs as above generators + batch_size
#Returns a generator that yields batches of data from random_generator
def make_batch(dataframe, image_path, batch_size=32, augment_batch=False, height=224, width=224, shuffle=True, augment=False, yieldY=True):
    if augment_batch:
        gen = generator(dataframe, image_path, height, width, shuffle, augment, yieldY)
    else:
        gen = random_generator(dataframe, image_path, height, width, shuffle, augment, yieldY)
    while True:
        x_train = np.empty([0, height, width, 3], dtype=np.float32)
        y_train = np.empty([0], dtype=np.int32)
        for image in range(batch_size):
            [x_data, y_data] = next(gen)
            x_train = np.append(x_train, x_data, axis=0)
            y_train = np.append(y_train, y_data, axis=0)
        yield [x_train, y_train]

#Takes a dataframe of objects and y-variable and returns the number of images with valid y-variable data
def numImgs(dataframe):
    y_name = 'year'
    count = 0
    for i in range(len(dataframe)):
        try:
            if not pd.isna(dataframe.iloc[i].loc[y_name]):
                yearvalue = float(dataframe.iloc[i].loc[y_name])
                pics = dataframe.iloc[i].loc['picIDs'].split(',')
                count += len(pics)
        except:
            continue
    return count

#True with probability prop and False with probability 1-prop
def coin_flip(prop):
    r = random.random()
    if r <= prop:
        return True 
    else:
        return False


#Session variables
experiment_path = '.'
train_image_path = '../../bjpics' #bjpics, chinapics
traindf = pd.read_csv(os.path.join(experiment_path, 'qingciqi_train.txt')) #'pbjyear_train.txt'))
val_image_path = '../../bjpics'
valdf = pd.read_csv(os.path.join(experiment_path, 'qingciqi_test.txt')) #'pbjyear_test.txt'))
epoch_number = 1000
do_batchTrain = True
augment_batch = True
batch_size = 16
img_height = 224
img_width = 224
model_name = 'resnet' #vgg16, resnet, inception, load
model_save_name = 'qingciqi_resnet2_feature_extraction'
model_load = 'pbjyear_mq_vgg16_feature_extraction3'
learning_rate = 1e-5
decay = 1e-3
momentum = 9e-1
verbose = 1
regularization = 0.1
checkpoint = ModelCheckpoint(os.path.join(experiment_path, './'+model_save_name+'/'+model_save_name+'.h5'), monitor='val_loss', verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=25)
callbacks = [checkpoint]

#Generators
train_generator = generator(dataframe=traindf,
                            image_path=train_image_path,
                            height=img_height,
                            width=img_width,
                            shuffle=True,
                            augment=True,
                            yieldY=True
                            )

train_batch_generator = make_batch(dataframe=traindf,
                                   image_path=train_image_path,
                                   batch_size=batch_size,
                                   augment_batch=augment_batch,
                                   height=img_height,
                                   width=img_width,
                                   shuffle=True,
                                   augment=True,
                                   yieldY=True
                                   )

val_generator = generator(dataframe=valdf,
                          image_path=val_image_path,
                          height=img_height,
                          width=img_width,
                          shuffle=False,
                          augment=False,
                          yieldY=True
                          )

#Model
if model_name == 'vgg16':

    net_input = Input(shape=(img_height, img_width, 3))
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3))(net_input)
    conv_base.trainable = False
    flatten1 = Flatten()(conv_base)
    dropout1 = Dropout(0.5)(flatten1)
    dense1 = Dense(
        1,
        activation='relu',
        kernel_regularizer=regularizers.l2(regularization))(dropout1)
    with tf.device('/cpu:0'):
        model = Model(inputs=net_input, outputs=dense1)

elif model_name == 'resnet':
    
    net_input = Input(shape=(img_height, img_width, 3))
    conv_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3))(net_input)
    conv_base.trainable = False
    flatten1 = Flatten()(conv_base)
    dropout1 = Dropout(0.5)(flatten1)
    dense1 = Dense(
        1,
        activation='relu',
        kernel_regularizer=regularizers.l2(regularization))(dropout1)
    with tf.device('/cpu:0'):
        model = Model(inputs=net_input, outputs=dense1)
    
    
elif model_name == 'inception':
    #model = inception_v1(shape=(img_height, img_width, 3), num_classes=1)
    net_input = Input(shape=(img_height, img_width, 3))
    conv_base = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3))(net_input)
    conv_base.trainable = False
    flatten1 = Flatten()(conv_base)
    dropout1 = Dropout(0.5)(flatten1)
    dense1 = Dense(
        1,
        activation='relu',
        kernel_regularizer=regularizers.l2(regularization))(dropout1)
    with tf.device('/cpu:0'):
        model = Model(inputs=net_input, outputs=dense1)

else:
    with tf.device('/cpu:0'):
        model = load_model(os.path.join(experiment_path, './'+model_load+'/'+model_load+'.h5'))

multi_model = multi_gpu_model(model, gpus=2)
model.summary()

numTrainImgs = numImgs(traindf)
numValImgs = numImgs(valdf)

#Train
'''
learning_rates = list((1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))
decays = list((1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))

for learning_rate in learning_rates:
    for decay in decays:
        print('Learning rate: ' + str(learning_rate) + '\nDecay: ' + str(decay))
        if not model_name == 'load':
            model.compile(optimizers.rmsprop(lr=learning_rate, decay=decay),loss='categorical_crossentropy',metrics=['accuracy'])
        
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=numTrainImgs,
                                      validation_data=val_generator,
                                      validation_steps=numValImgs,
                                      epochs=epoch_number,
                                      verbose=2
        )
'''
if not model_name == 'load':
    multi_model.compile(optimizers.rmsprop(lr=learning_rate, decay=decay), loss='mean_absolute_percentage_error')
    #multi_model.compile(optimizers.rmsprop(lr=learning_rate, decay=decay), loss='mean_squared_error')
    #multi_model.compile(optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum), loss='mean_absolute_percentage_error')

#Model folder
try:
    os.mkdir(os.path.join(experiment_path, model_save_name))
except:
    print('Saving over model ' + model_save_name)

if not do_batchTrain:
    history = multi_model.fit_generator(generator=train_generator,
                                        steps_per_epoch=numTrainImgs,
                                        validation_data=val_generator,
                                        validation_steps=numValImgs,
                                        epochs=epoch_number,
                                        verbose=verbose,
                                        callbacks=callbacks
                                        )

else:
    history = multi_model.fit_generator(generator=train_batch_generator,
                                        steps_per_epoch=numTrainImgs//batch_size,
                                        validation_data=val_generator,
                                        validation_steps=numValImgs,
                                        epochs=epoch_number,
                                        verbose=verbose,
                                        callbacks=callbacks
                                        )

#Save model parameters
model.save(os.path.join(experiment_path, './'+model_save_name+'/'+model_save_name+'.h5'))

#Plots
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r*', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(experiment_path, './'+model_save_name+'/'+'loss.png'))
