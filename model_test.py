# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:22:29 2019

@author: weiho
"""

import keras
from keras import backend as K
from keras.applications import VGG16, ResNet50
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model
from keras import regularizers, optimizers
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report

from aug_img_generator import generator, numImgs, to_categorical, update_dictionary, coin_flip, sort_list_by_index, get_freq_dict, get_freq_list

#Session variables
experiment_path = '.'
image_path = '../../bjpics' #bjpics, chinapics
testdf = pd.read_csv(os.path.join(experiment_path, 'largetype_test.txt'))
y_name = 'typeName'
model_test_name = 'largetype_inception_feature_extraction'
labelindices = pd.read_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+'.txt'))
classes = list(labelindices.loc[:, 'class'])

y_labels = np.empty([0], dtype=np.int32)

for i in range(len(testdf)):
    if testdf.iloc[i].loc[y_name] in classes:
        for pic in testdf.iloc[i].loc['picIDs'].split(','):
            y_labels = np.append(y_labels, [testdf.iloc[i].loc[y_name]])

y_truths = to_categorical(y_labels, classes)

print('Classes in model: ' + str(classes))

test_generator = generator(dataframe=testdf,
                           image_path=image_path,
                           y_name=y_name,
                           y_classes=set(classes), 
                           height=224,
                           width=224,
                           shuffle=False,
                           augment=False,
                           yieldY=False)

totalExamples = numImgs(testdf, y_name, y_classes=set(testdf.loc[:, y_name]))
labeledExamples = numImgs(testdf, y_name, y_classes=classes)

#Load model
model = load_model(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+'.h5'))

testresults = model.predict_generator(generator=test_generator,
                                    steps=totalExamples,
                                    verbose=1)

#Write all test results

#Probabilities of each class by image
guessprobs = pd.DataFrame(testresults, index=None)
guessprobs.columns = classes

#Highest probability class by image
guessesdf = np.empty([0], dtype=np.int32)
for i in range(len(testresults)):
    guessesdf = np.append(guessesdf, classes[np.argmax(testresults[i])])
guessesdf = pd.DataFrame(guessesdf, columns=['pred'], index=None)

#PictureIDs, objectID, categoryNames, and true labels by image
objpics = np.empty([0, 4], dtype=np.int32)
for i in range(len(testdf)):
    for pic in testdf.iloc[i].loc['picIDs'].split(','):
        objpics = np.append(objpics, [[pic, testdf.iloc[i].loc['objectID'], testdf.iloc[i].loc['categoryName'], testdf.iloc[i].loc[y_name]]], axis=0)
objpicsdf = pd.DataFrame(objpics, columns=['picID', 'objectID', 'categoryName', 'truth'], index=None)

resultsdf = objpicsdf.join(guessesdf.join(guessprobs))
resultsdf.to_csv(os.path.join(experiment_path, './'+model_test_name+'/'+model_test_name+':test_results.txt'), encoding='utf-8', index=None, na_rep='NA')

#Top k-accuracies by individual image
k_accuracies = []
for k in range(1, len(classes)+1):
    k_correct = 0
    for i in range(len(resultsdf)):
        pic_truth = resultsdf.iloc[i].loc['truth']
        if pic_truth in classes:
            try:
                guessprob = list(guessprobs.iloc[i])
                guessorder = sort_list_by_index(guessprob)
                if classes.index(pic_truth) in guessorder[:k]:
                    k_correct += 1
            except:
                print(pic_truth + ' label not in training set')
    k_correct /= labeledExamples
    k_accuracies.append(k_correct)
    print(str(k) + '-accuracy by image: ' + str(k_correct))

#Confusion matrix
#[i, j] = n : label j is the nth most likely label given by the model to images with true label i
confusion_matrix = np.zeros((len(classes), len(classes)))
for i in range(len(resultsdf)):
    pic_truth = resultsdf.iloc[i].loc['truth']
    if pic_truth in classes:
        try:
            truthindex = classes.index(pic_truth)
            rankprobs = sort_list_by_index(list(guessprobs.iloc[i]))
            for guess in range(len(rankprobs)):
                confusion_matrix[truthindex, rankprobs[guess]] += guess+1
        except:
            print(pic_truth + ' label not in training set')

results_freq_dict = get_freq_dict(resultsdf, 'truth', classes)
for i in range(confusion_matrix.shape[0]):
    if classes[i] in results_freq_dict.keys():
        confusion_matrix[i] /= results_freq_dict[classes[i]]
    else:
        confusion_matrix[i] = [np.nan]*len(classes)

#Top k-accuracies by object
#Top k guesses per image are totaled and the top k totals are considered
k_objectaccs = []
for k in range(1, len(classes)+1):
    k_correct = 0
    objtotal = 0
    picindex = 0
    for i in range(len(testdf)):
        objname = testdf.iloc[i].loc[y_name]
        if objname in classes:
            numpics = len(testdf.iloc[i].loc['picIDs'].split(','))
            guesses = [0]*len(classes)
            for j in range(numpics):
                picguesses = sort_list_by_index(list(guessprobs.iloc[picindex]))
                for x in range(k):
                    guesses[picguesses[x]] += 1
                picindex += 1
            objguesses = sort_list_by_index(guesses)
            if classes.index(objname) in objguesses[:k]:
                k_correct += 1
            objtotal += 1
        else:
            picindex += len(testdf.iloc[i].loc['picIDs'].split(','))
    k_correct /= objtotal
    k_objectaccs.append(k_correct)
    print(str(k) + '-accuracy by object: ' + str(k_correct))

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
    
    #k-accuracies per object
    summary_writefile.write('k-accuracies per object: \n')
    for k in range(len(k_objectaccs)):
        summary_writefile.write(str(k+1) + '-accuracy: ' + str(k_objectaccs[k]) + '\n')
    summary_writefile.write('\n')
