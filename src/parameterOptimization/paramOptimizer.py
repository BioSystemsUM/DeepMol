#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:16:31 2019

@author: jcorreia
"""


################################################

# OLD FILE -- ONLY FOR REFERENCE

# REMOVE LATER

#################################################


import numpy as np
import pandas as pd
#from fingerprintGenerator import fingerprintGenerator
#from embeddingsGenerator import embeddingsGenerator
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau

from time import time
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers

class paramOptimizer():
    
    """
    Class to optimize model parameters
    
    """
    
    def __init__(self, modelType, optType, paramDic, dataX, datay, n_jobs = -1):
        self.modelType = modelType
        self.optType = optType
        self.paramDic = paramDic
        self.dataX = dataX
        self.datay = datay
        self.n_jobs = n_jobs
        self.model = None
    
    
    def _rfClassifier(self):
        # build a classifier
        clf = RandomForestClassifier()
        # Look at parameters used by our current forest
        print('Parameters currently in use:\n')
        pprint(clf.get_params())
        return clf

    def _svmClassifier(self):
        clf = SVC()
        # Look at parameters used by our current forest
        print('Parameters currently in use:\n')
        pprint(clf.get_params())
        return clf
    
    
    def _base_DNN_model(self, units1 = 512, units2 = 512, dropout_rate=0.0, hidden_layers=1, l1 = 0, l2 = 0, 
                        optimizer = 'adam', batchNormalization = True):
        # create model
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
        #Reduce lr callback 
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
        return model
    
    
        
    
    # Utility function to report best scores
    def _report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
                
                
    def _randomizedSearch(self, x, y, param_dic, nfolds = 5, n_iter_search = 15):

        #  %%%%%%%%%%%%%%%%   RANDOMIZED SEARCH   %%%%%%%%%%%%%%%%%%%%%%%%
        
        # run randomized search
        random_search = RandomizedSearchCV(self.model, param_distributions=param_dic,
                                           n_iter=n_iter_search, cv=nfolds, n_jobs = self.n_jobs)
    
        start = time()
        random_search.fit(x.values, y.values)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        self._report(random_search.cv_results_)
        return random_search.best_params_
        
    def _gridSearch(self, x, y, param_dic, nfolds = 5):
        #  %%%%%%%%%%%%%%%%   GRID SEARCH   %%%%%%%%%%%%%%%%%%%%%%%%
    
        # run grid search
        grid_search = GridSearchCV(self.model, param_grid=param_dic, cv=nfolds, n_jobs = self.n_jobs)
        start = time()
        grid_search.fit(x.values, y.values)
    
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.cv_results_['params'])))
        self._report(grid_search.cv_results_)
        
        return grid_search.best_params_
    
    def getOptParams(self):
        if self.modelType == 'rf':
            self.model = self._rfClassifier()
        elif self.modelType == 'svm':
            self.model = self._svmClassifier()
        elif self.modelType == 'dnn':
            print(':)')
            self.model = KerasClassifier(build_fn=self._base_DNN_model, verbose=0)
        else :
            print('Invalid model type!')
            return None
        if self.optType == 'gridSearch':
            return self._gridSearch(self.dataX, self.datay, self.paramDic)
        elif self.optType == 'randomizedSearch':
            return self._randomizedSearch(self.dataX, self.datay, self.paramDic)
        else :
            print('Invalid optimization type!')
            return None 


if __name__ == '__main__':
    df = pd.read_csv('merged_dataset_final2.csv', sep = ';', header = 0)
    mrg_fps = fingerprintGenerator(df)
    mrg_dataset = mrg_fps.getFingerprintsDataset()
    fps_df = mrg_dataset.dropna()
    fps_y = fps_df['Class']
    fps_x = fps_df.drop(columns = ['Smiles', 'Class'])
    
    param_dist_rf = {"max_depth": [10, 50, 80, None], 
                  "max_features": ['auto', 'sqrt'],   
                  "min_samples_split": [int(x) for x in range(2, 12, 3)], 
                  "bootstrap": [True, False],         
                  "criterion": ["gini", "entropy"],
                  "n_estimators" : [int(x) for x in range(100, 1000, 100)]}
    
    optParams = paramOptimizer('rf', 'gridSearch', param_dist_rf, fps_x, fps_y).getOptParams()