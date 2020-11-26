#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:18:27 2019

@author: jcorreia
"""


################################################

# OLD FILE -- ONLY FOR REFERENCE

# REMOVE LATER

#################################################



import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.models import model_from_json
import matplotlib.pyplot as plt

        
def trainRF(X_train, y_train, parameterValues, nfolds = 5):
    
    rf = RandomForestClassifier(**parameterValues, n_jobs = -1).fit(X_train,y_train)
    
    scores = cross_val_score(rf, X_train, y_train, cv=nfolds, n_jobs = -1)
    print(scores)
    print("CV accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return rf


def trainSVM( X_train, y_train, parameterValues, nfolds = 5):
    
    svmc = svm.SVC(**parameterValues, probability = True).fit(X_train,y_train)
    
    scores = cross_val_score(svmc, X_train, y_train, cv=nfolds, n_jobs = -1)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    return svmc

def trainDNN(X_train, y_train, X_test, y_test, parameterValues):
    def create_model(units1 = 512, units2 = 512, dropout_rate=0.0, hidden_layers=1, l1 = 0, l2 = 0, 
                        optimizer = 'adam', batchNormalization = True):
        # create model
        print('dropout_rate = ', dropout_rate)
        model = Sequential()
        model.add(Dense(units=units1, activation="relu"))
        if batchNormalization:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        for i in range(hidden_layers):
            model.add(Dense(units=units2, activation="relu", kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)))
            if batchNormalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        ##Compile model and make it ready for optimization
        model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

        return model
    
    print('=== DNN ===')
    model = KerasClassifier(build_fn=create_model, **parameterValues, verbose=0)

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
    
    #callbacks = [es, reduce_lr]
    callbacks = [reduce_lr]
    #Training
    dnn = model.fit(X_train, y_train, nb_epoch=500, batch_size=10, callbacks=callbacks,
                    validation_data = (X_test, y_test))

    print('Training Accuracy: ', np.mean(dnn.history['acc']))
    print('Validation Accuracy: ', np.mean(dnn.history['val_acc']))
    
    return dnn


def saveModel(modelType, model, name):
    if modelType != 'dnn':
        pickle.dump(model, open(name, 'wb'))
    else : 
        model.model.save(name + '.h5')
        
        
if __name__ == '__main__':
    pass