# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:21:52 2019

@author: weiho
"""
import os
import pandas as pd
import statistics

from train_val_split import train_val_split
from aux_func import split_objects
from categorical import train_categorical, eval_categorical, predict_categorical
from regression import train_regression, eval_regression, predict_regression

main_experiment_path = '../experiments'

def eval_CNN_categorical(experiment_name, trials, inFile, image_path, categoryName, numCat, gpus, model_type, epoch_number, augment_number, learning_rate, decay, regularization, batch_size, img_height, img_width):
    
    #Experiment folder
    try:
        os.mkdir(os.path.join(main_experiment_path, experiment_name))
    except:
        print('Saving over experiment ' + experiment_name)
        
    top_1_accuracies = list()
    
    for i in range(1, len(trials)+1):
        
        trial_path = os.path.join(main_experiment_path, experiment_name, experiment_name+'_'+str(i))
        
        #Trial i folder
        try:
            os.mkdir(trial_path)
        except:
            print('Saving over trial ' + str(i) + ' of experiment ' + experiment_name)
            
        y_classes = train_val_split(inFile, outFile=os.path.join(trial_path, 'df'), categoryName=categoryName, numCat=numCat, val_prop=0.3)
        
        traindf = pd.read_csv(os.path.join(trial_path, 'df_train.txt'))
        valdf = pd.read_csv(os.path.join(trial_path, 'df_val.txt'))
        
        split_objects(traindf, os.path.join(trial_path, 'splitdf_train.txt'))
        split_objects(valdf, os.path.join(trial_path, 'splitdf_val.txt'))
        
        train_path = os.path.join(trial_path, 'splitdf_train.txt')
        val_path = os.path.join(trial_path, 'splitdf_val.txt')
        
        #Train model i
        train_categorical(gpus=gpus,
                          experiment_path=trial_path,
                          train_df_path=train_path,
                          train_image_path=image_path,
                          val_df_path=val_path,
                          val_image_path=image_path,
                          y_name=categoryName,
                          y_classes=y_classes,
                          augment_number=augment_number,
                          model_type=model_type,
                          model_save_name=experiment_name,
                          epoch_number=epoch_number,
                          learning_rate=learning_rate,
                          decay=decay,
                          regularization=regularization,
                          batch_size=batch_size,
                          img_height=img_height,
                          img_width=img_width
                          )
        
        #Evaluate model i
        k_accuracies = eval_categorical(experiment_path=trial_path,
                                        test_df_path=val_path,
                                        test_image_path=image_path,
                                        y_name=categoryName,
                                        model_test_name=experiment_name,
                                        height=img_height,
                                        width=img_width
                                        )
        
        #Top 1 accuracy
        top_1_accuracies.append(k_accuracies[0])
        
    #Mean and variance of top_1 accuracies
    mean = statistics.mean(top_1_accuracies)
    variance = statistics.variance(top_1_accuracies)
    
    #Write summary of test results
    with open(os.path.join(main_experiment_path, experiment_name, experiment_name+':experiment_summary.txt'), 'w') as summary_writefile:
        summary_writefile.write('Prediction error mean in years: ' + mean + '\n')
        summary_writefile.write('Prediction error variance in years: ' + variance + '\n')
        for i in range(1, len(top_1_accuracies)+1):
            summary_writefile.write('Accuracy of model ' + str(i) + ': ' + top_1_accuracies[i] + '\n')
        
    #Return test results summary
    return {'mean': mean,
            'variance': variance}
        