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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report, confusion_matrix
import sys

from aug_img_generator import generator, make_batch, numImgs, get_freq_dict, get_freq_list

from inception_model import inception_v1
from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf

#Session variables
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
checkpoint = ModelCheckpoint(os.path.join(experiment_path, './'+model_save_name+'/'+model_save_name+'.h5'), monitor='val_loss', verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=25)
callbacks = [checkpoint]

#Runtime variables
freq_dict = get_freq_dict(traindf, y_name, y_classes)
freq_list = get_freq_list(traindf, y_name, y_classes)
num_labels = len(freq_list)

class_weight = list()
for freq in freq_list:
    class_weight.append(freq_dict[freq_list[-1]] / freq_dict[freq])

#Generators
train_generator = generator(dataframe=traindf,
                            image_path=train_image_path,
                            y_name=y_name,
                            y_classes=y_classes,
                            height=img_height,
                            width=img_width,
                            shuffle=True,
                            augment=True,
                            yieldY=True
                            )

train_batch_generator = make_batch(dataframe=traindf,
                                   image_path=train_image_path,
                                   y_name=y_name,
                                   y_classes=y_classes,
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
                          y_name=y_name,
                          y_classes=y_classes,
                          height=img_height,
                          width=img_width,
                          shuffle=False,
                          augment=True,
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
    #model = inception_v1(shape=(img_height, img_width, 3), num_classes=num_labels)
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
    with tf.device('/cpu:0'):
        model = load_model(os.path.join(experiment_path, './'+model_load+'/'+model_load+'.h5'))

multi_model = multi_gpu_model(model, gpus=2)
model.summary()

numTrainImgs = numImgs(traindf, y_name, y_classes=y_classes)
numValImgs = numImgs(valdf, y_name, y_classes=y_classes)

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
                                      class_weight=class_weight,
                                      epochs=epoch_number,
                                      verbose=verbose
        )
'''
if not model_name == 'load':
    #multi_model.compile(optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy']) 
    multi_model.compile(optimizers.rmsprop(lr=learning_rate, decay=decay), loss='categorical_crossentropy', metrics=['accuracy'])

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
                                        class_weight=class_weight,
                                        epochs=epoch_number,
                                        verbose=verbose,
                                        callbacks=callbacks
                                        )

else:
    history = multi_model.fit_generator(generator=train_batch_generator,
                                        steps_per_epoch=numTrainImgs//batch_size,
                                        validation_data=val_generator,
                                        validation_steps=numValImgs,
                                        class_weight=class_weight,
                                        epochs=epoch_number,
                                        verbose=verbose,
                                        callbacks=callbacks
                                        )

#Save model parameters
model.save(os.path.join(experiment_path, './'+model_save_name+'/'+model_save_name+'.h5'))

#Save model index meanings
with open(os.path.join(experiment_path, './'+model_save_name+'/'+model_save_name+'.txt'), mode='w', encoding='utf-8') as filewrite:
    csv_writer = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['class', 'freq', 'index'])
    for i in range(len(freq_list)):
        csv_writer.writerow([freq_list[i], freq_dict[freq_list[i]], i])

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
plt.legend()
plt.savefig(os.path.join(experiment_path, './'+model_save_name+'/'+'accuracy.png'))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r*', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(experiment_path, './'+model_save_name+'/'+'loss.png'))
