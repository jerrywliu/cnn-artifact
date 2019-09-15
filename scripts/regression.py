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
import statistics
import tensorflow as tf

from generators import regression_generator, make_batch, test_generator
from aux_func import allImgs, get_freq_dict

def train_regression(gpus, experiment_path, train_df_path, train_image_path, val_df_path, val_image_path, y_name, augment_number,
                     model_type, model_save_name, epoch_number, learning_rate, decay, regularization, batch_size=32, img_height=224, img_width=224):


    checkpoint = ModelCheckpoint(os.path.join(experiment_path, './'+model_save_name+'/'+model_save_name+'.h5'), monitor='val_loss', verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=25)
    callbacks = [checkpoint]

    #Session variables
    traindf = pd.read_csv(train_df_path)
    valdf = pd.read_csv(val_df_path)
    
    numTrainImgs = allImgs(traindf, y_name, countNAs=False)
    numValImgs = allImgs(valdf, y_name, countNAs=False)
    
    #Generators
    train_generator = make_batch(generator='regression',
                                 dataframe=traindf,
                                 image_path=train_image_path,
                                 y_name=y_name,
                                 batch_size=batch_size,
                                 augment_number=augment_number,
                                 height=img_height,
                                 width=img_width,
                                 forTrain=True
                                 )
    
    val_generator = make_batch(generator='regression',
                               dataframe=valdf,
                               image_path=val_image_path,
                               y_name=y_name,
                               batch_size=batch_size,
                               augment_number=augment_number,
                               height=img_height,
                               width=img_width,
                               forTrain=False
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
        print('\nError in train_regression: invalid model type\n')

    multi_model = multi_gpu_model(model, gpus=gpus)
    model.summary()

    #Model folder
    try:
        os.mkdir(os.path.join(experiment_path, model_save_name))
    except:
        print('Saving over model ' + model_save_name)

    #Train
    multi_model.compile(optimizers.RMSprop(lr=learning_rate, decay=decay), loss='mean_squared_error')

    history = multi_model.fit_generator(generator=train_generator,
                                        steps_per_epoch=numTrainImgs//batch_size,
                                        validation_data=val_generator,
                                        validation_steps=numValImgs,
                                        epochs=epoch_number,
                                        verbose=verbose,
                                        callbacks=callbacks
                                        )

    #Save model parameters
    model.save(os.path.join(experiment_path, model_save_name+'.h5'))
    
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
    plt.savefig(os.path.join(experiment_path, 'loss.png'))

#experiment_path/model_test_name/model_test_name.h5 = model being evaluated
#Evaluates performance of model_test_name on test_df_path
#Creates test results file at experiment_path/model_test_name/model_test_name:test_results.txt
#Creates summary of test results at experiment_path/model_test_name/model_test_name:test_summary.txt
#Returns dictionary containing mean and variance of prediction error
def eval_regression(experiment_path, test_df_path, test_image_path, y_name, model_test_name, height=224, width=224):

    #Session variables
    testdf = pd.read_csv(test_df_path)

    #PictureIDs, objectID, categoryNames, and true labels by image
    objpics = np.empty([0, 4], dtype=np.int32)
    for i in range(len(testdf)):
        if not pd.isna(testdf.iloc[i].loc[y_name]):
            for pic in testdf.iloc[i].loc['picIDs'].split(','):
                objpics = np.append(objpics, [[pic, testdf.iloc[i].loc['objectID'], testdf.iloc[i].loc['categoryName'], testdf.iloc[i].loc[y_name]]], axis=0)
    objpicsdf = pd.DataFrame(objpics, columns=['picID', 'objectID', 'categoryName', 'truth'], index=None)
    
    #Load model
    model = load_model(os.path.join(experiment_path, model_test_name+'.h5'))

    #Test generator
    eval_generator = test_generator(dataframe=testdf,
                                    image_path=test_image_path,
                                    y_name=y_name,
                                    y_classes=None,
                                    include_NA=False,
                                    height=height,
                                    width=width
                                    )
    
    labeledExamples = allImgs(testdf, y_name, include_NA=False)
    
    testresults = model.predict_generator(generator=eval_generator,
                                          steps=labeledExamples,
                                          verbose=1)
    
    #Year predicted for each image
    guessyears = pd.DataFrame(testresults, columns=['pred'], index=None)
    
    #Errors
    errors = np.array(guessyears.loc[:, 'pred'])-np.array(objpicsdf.loc[:, 'truth'])
    mean = statistics.mean(errors)
    variance = statistics.variance(errors)
    
    #Test results file
    resultsdf = objpicsdf.join(guessyears)
    resultsdf.to_csv(os.path.join(experiment_path, model_test_name+':test_results.txt'), encoding='utf-8', index=None, na_rep='NA')
    
    #Write summary of test results
    with open(os.path.join(experiment_path, model_test_name+':test_summary.txt'), 'w') as summary_writefile:
        summary_writefile.write('Prediction error mean in years: ' + mean + '\n')
        summary_writefile.write('Prediction error variance in years: ' + variance + '\n')
        
    #Return test results summary
    return {'mean': mean,
            'variance': variance}
        
#experiment_path/model_test_name/model_test_name.h5 = model being evaluated
#Uses model_test_name to predict years of images in test_df_path
#Creates test results file at experiment_path/model_test_name/model_test_name:results_save_name.txt
def predict_regression(experiment_path, test_df_path, test_image_path, model_test_name, results_save_name, height=224, width=224):
    
    #Session variables
    testdf = pd.read_csv(test_df_path)
    
    #PictureIDs and objectID for each image
    objpics = np.empty([0, 2], dtype=np.int32)
    for i in range(len(testdf)):
        for pic in testdf.iloc[i].loc['picIDs'].split(','):
            objpics = np.append(objpics, [[pic, testdf.iloc[i].loc['objectID']]], axis=0)
    objpicsdf = pd.DataFrame(objpics, columns=['picID', 'objectID'], index=None)
    
    #Load model
    model = load_model(os.path.join(experiment_path, model_test_name+'.h5'))

    #Test generator
    predict_generator = test_generator(dataframe=testdf,
                                    image_path=test_image_path,
                                    y_name=None,
                                    y_classes=None,
                                    include_NA=True,
                                    height=height,
                                    width=width
                                    )
    
    totalExamples = allImgs(testdf, y_name=None, countNAs=True)
    
    testresults = model.predict_generator(generator=predict_generator,
                                          steps=totalExamples,
                                          verbose=1)
    
    #Year predicted for each image
    guessyears = pd.DataFrame(testresults, columns=['pred'], index=None)
    
    #Test results file
    resultsdf = objpicsdf.join(guessyears)
    resultsdf.to_csv(os.path.join(experiment_path, model_test_name+':' + results_save_name + '.txt'), encoding='utf-8', index=None, na_rep='NA')
    
