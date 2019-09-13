# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:16:45 2019
@author: weiho
"""
import os
'''
#Uses GPU 0
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
'''
import csv
import keras
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model
from keras import regularizers, optimizers
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import tensorflow as tf

from generators import categorical_generator, make_batch, test_generator
from aux_func import numImgs, allImgs, get_freq_dict

experiment_path = '.'
train_image_path = '../../bjpics' #bjpics, chinapics
traindf = pd.read_csv(os.path.join(experiment_path, 'qingciqiper_train.txt')) #cat, type
val_image_path = '../../bjpics'
valdf = pd.read_csv(os.path.join(experiment_path, 'qingciqiper_test.txt'))
y_name = 'emperor'
y_classes = set(traindf.loc[:, y_name])
if np.nan in y_classes:
    y_classes.remove(np.nan)
epoch_number = 500
do_batchTrain = True
augment_batch = True
batch_size = 32
img_height = 224
img_width = 224
model_name = 'inception' #vgg16, resnet, inception, load
model_save_name = 'qingciqiper_inception_feature_extraction'
model_load = 'pbjtype_inception_feature_extraction'
learning_rate = 1e-5
decay = 1e-3
momentum = 0.9
verbose = 1
regularization = 0.1

#model_type = vgg16, resnet, inception
#y_classes = categories used during training
#Trains a model on data from train_df_path, cross-validating with val_df_path
#Saves model at experiment_path/model_save_name.h5, class indices at experiment_path/model_save_name:indices.txt
#Accuracy at experiment_path/accuracy.png, loss at experiment_path/loss.png
def train_categorical(gpus, experiment_path, train_df_path, train_image_path, val_df_path, val_image_path, y_name, y_classes, augment_number,
                      model_type, model_save_name, epoch_number, learning_rate, decay, regularization, batch_size=32, img_height=224, img_width=224):
    
    checkpoint = ModelCheckpoint(os.path.join(experiment_path, model_save_name+'.h5'), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=25)
    callbacks = [checkpoint]
    
    #Session variables
    traindf = pd.read_csv(train_df_path)
    valdf = pd.read_csv(val_df_path)
    
    freq_dict = get_freq_dict(traindf, y_name, y_classes)
    num_labels = len(y_classes)
    
    numTrainImgs = numImgs(traindf, y_name, y_classes)
    numValImgs = numImgs(valdf, y_name, y_classes)
    
    class_weight = list(map(lambda x: 1/x, y_classes))
    
    #Generators
    train_generator = make_batch(generator='categorical',
                                 dataframe=traindf,
                                 image_path=train_image_path,
                                 y_name=y_name,
                                 y_classes=y_classes,
                                 batch_size=batch_size,
                                 augment_number=augment_number,
                                 height=img_height,
                                 width=img_width,
                                 forTrain=True
                                 )
    
    val_generator = make_batch(generator='categorical',
                               dataframe=valdf,
                               image_path=val_image_path,
                               y_name=y_name,
                               y_classes=y_classes,
                               batch_size=batch_size,
                               height=img_height,
                               width=img_width,
                               forTrain=False
                               )
    
    #Model
    if model_type == 'vgg16':
    
        net_input = Input(shape=(img_height, img_width, 3))
        conv_base = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(img_height, img_width, 3))(net_input)
        conv_base.trainable = False
        flatten1 = Flatten()(conv_base)
        dropout1 = Dropout(0.5)(flatten1)
        dense1 = Dense(
            num_labels,
            activation='softmax',
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
            num_labels,
            activation='softmax',
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
            num_labels,
            activation='softmax',
            kernel_regularizer=regularizers.l2(regularization))(dropout1)
        with tf.device('/cpu:0'):
            model = Model(inputs=net_input, outputs=dense1)
    
    else:
        print('\nError in train_categorical: invalid model type\n')
    
    multi_model = multi_gpu_model(model, gpus=gpus)
    model.summary()
        
    #Train
    multi_model.compile(optimizers.RMSprop(lr=learning_rate, decay=decay), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = multi_model.fit_generator(generator=train_generator,
                                        steps_per_epoch=numTrainImgs,
                                        validation_data=val_generator,
                                        validation_steps=numValImgs,
                                        class_weight=class_weight,
                                        epochs=epoch_number,
                                        verbose=1,
                                        callbacks=callbacks
                                        )
    
    #Save model parameters
    model.save(os.path.join(experiment_path, model_save_name+'.h5'))
    
    #Save model index meanings
    with open(os.path.join(experiment_path, model_save_name+':indices.txt'), mode='w', encoding='utf-8') as filewrite:
        csv_writer = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['class', 'frequency', 'index'])
        for i in range(len(y_classes)):
            csv_writer.writerow([y_classes[i], freq_dict[y_classes[i]], i])
    
    #Plots
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r*', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(experiment_path, 'accuracy.png'))
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r*', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(experiment_path, 'loss.png'))

#Session variables
experiment_path = '.'
image_path = '../../bjpics' #bjpics, chinapics
testdf = pd.read_csv(os.path.join(experiment_path, 'largetype_test.txt'))
y_name = 'typeName'
model_test_name = 'largetype_inception_feature_extraction'
labelindices = pd.read_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+'.txt'))
classes = list(labelindices.loc[:, 'class'])

#experiment_path/model_test_name/model_test_name.h5 = model being evaluated, experiment_path/model_test_name/model_test_name:indices.txt = class indices
#Evaluates performance of model_test_name on test_df_path
#Creates test results file at experiment_path/model_test_name/model_test_name:test_results.txt
#Creates summary of test results at experiment_path/model_test_name/model_test_name:test_summary.txt
#Returns accuracies by class
def eval_categorical(experiment_path, test_df_path, test_image_path, y_name, model_test_name, height=224, width=224):

    #Session variables
    testdf = pd.read_csv(test_df_path)
    labelindices = pd.read_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+':indices.txt'))
    y_classes = list(labelindices.loc[:, 'class'])
    print('Classes in model ' + model_test_name + ': ' + str(y_classes))
    freq_dict = get_freq_dict(testdf, y_name, y_classes)

    #PictureIDs, objectID, categoryNames, and true labels by image
    objpics = np.empty([0, 4], dtype=np.int32)
    for i in range(len(testdf)):
        if testdf.iloc[i].loc[y_name] in y_classes:
            for pic in testdf.iloc[i].loc['picIDs'].split(','):
                objpics = np.append(objpics, [[pic, testdf.iloc[i].loc['objectID'], testdf.iloc[i].loc['categoryName'], testdf.iloc[i].loc[y_name]]], axis=0)
    objpicsdf = pd.DataFrame(objpics, columns=['picID', 'objectID', 'categoryName', 'truth'], index=None)
    
    #Load model
    model = load_model(os.path.join(experiment_path, model_test_name+'.h5'))

    #Test generator
    eval_generator = test_generator(dataframe=testdf,
                                    image_path=test_image_path,
                                    y_name=y_name,
                                    y_classes=y_classes,
                                    include_NA=False,
                                    height=height,
                                    width=width
                                    )
    
    labeledExamples = numImgs(testdf, y_name, y_classes=y_classes)
    
    testresults = model.predict_generator(generator=eval_generator,
                                          steps=labeledExamples,
                                          verbose=1)
    
    #Probabilities of each class by image
    guessprobs = pd.DataFrame(testresults, index=None)
    guessprobs.columns = y_classes
    
    #Highest probability class by image
    guessesdf = np.argmax(testresults, axis=1)
    guessesdf = pd.DataFrame(guessesdf, columns=['pred'], index=None)
    
    #Test results file
    resultsdf = objpicsdf.join(guessesdf.join(guessprobs))
    resultsdf.to_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+':test_results.txt'), encoding='utf-8', index=None, na_rep='NA')
    
    #Top k-accuracies by individual image
    k_accuracies = np.zeros(len(y_classes))
    
    #Confusion matrix
    #[i, j] = n : label j is the nth most likely label given by the model to images with true label i
    confusion_matrix = np.zeros((len(y_classes), len(y_classes)))
    
    for i in range(len(resultsdf)):
        pic_truth = resultsdf.iloc[i].loc['truth']
        truthindex = y_classes.index(pic_truth)
        guessprob = list(guessprobs.iloc[i])
        rankguesses = list(map(lambda x : guessprob.index(x), sorted(guessprob)))
        for guess in range(len(rankguesses)):
            confusion_matrix[truthindex, rankguesses[guess]] += guess+1
        for k in range(1, len(y_classes)+1):
            if y_classes.index(pic_truth) in rankguesses[:k]:
                k_accuracies[k-1] += 1
    
    k_accuracies /= labeledExamples
    for i in range(confusion_matrix.shape[0]):
        confusion_matrix[i] /= freq_dict[classes[i]]
        
    #Write summary of test results
    with open(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+':test_summary.txt'), 'w') as summary_writefile:
        #k-accuracies per image
        summary_writefile.write('k-accuracies per image: \n')
        for k in range(len(k_accuracies)):
            summary_writefile.write(str(k+1) + '-accuracy: ' + str(k_accuracies[k]) + '\n')
        summary_writefile.write('\n')
        
        #Confusion matrix
        summary_writefile.write('Confusion matrix: nth most likely label to be assigned\n')
        summary_writefile.write('Classes: ' + str(classes) + '\n')
        for i in range(confusion_matrix.shape[0]):
            summary_writefile.write(classes[i] + ': ' + str(confusion_matrix[i]) + '\n')
        summary_writefile.write('\n')
        
    #Return accuracies by class
    return k_accuracies
        
#experiment_path/model_test_name/model_test_name.h5 = model being evaluated, experiment_path/model_test_name/model_test_name:indices.txt = class indices
#Uses model_test_name to predict classes of images in test_df_path
#Creates test results file at experiment_path/model_test_name/model_test_name:results_save_name.txt
#Returns accuracies by class
def predict_categorical(experiment_path, test_df_path, test_image_path, model_test_name, results_save_name, height=224, width=224):
    
    #Session variables
    testdf = pd.read_csv(test_df_path)
    labelindices = pd.read_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+':indices.txt'))
    y_classes = list(labelindices.loc[:, 'class'])
    print('Classes in model ' + model_test_name + ': ' + str(y_classes))
    
    #PictureIDs and objectID for each image
    objpics = np.empty([0, 2], dtype=np.int32)
    for i in range(len(testdf)):
        for pic in testdf.iloc[i].loc['picIDs'].split(','):
            objpics = np.append(objpics, [[pic, testdf.iloc[i].loc['objectID']]], axis=0)
    objpicsdf = pd.DataFrame(objpics, columns=['picID', 'objectID'], index=None)
    
    #Load model
    model = load_model(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+'.h5'))

    #Test generator
    predict_generator = test_generator(dataframe=testdf,
                                    image_path=test_image_path,
                                    y_name=None,
                                    y_classes=None,
                                    include_NA=True,
                                    height=height,
                                    width=width
                                    )
    
    totalExamples = allImgs(testdf, y_name=None, include_NA=True)
    
    testresults = model.predict_generator(generator=predict_generator,
                                          steps=totalExamples,
                                          verbose=1)
    
    #Probabilities of each class by image
    guessprobs = pd.DataFrame(testresults, index=None)
    guessprobs.columns = y_classes
    
    #Highest probability class by image
    guessesdf = np.argmax(testresults, axis=1)
    guessesdf = pd.DataFrame(guessesdf, columns=['pred'], index=None)
    
    #Predict results file
    resultsdf = objpicsdf.join(guessesdf.join(guessprobs))
    resultsdf.to_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+':' + results_save_name + '.txt'), encoding='utf-8', index=None, na_rep='NA')
    