#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:38:41 2019

@author: jcorreia
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV, SelectFromModel, SelectPercentile
from sklearn.ensemble import RandomForestClassifier

class featureSelection():
    
    """
    Class for feature selection/dimensionality reduction
    
    """
    
    def __init__(self, features, labels, fsType, param = None):
        self.fsType = fsType
        self.param = param
        self.X = features
        self.y = labels
        self.column_indexes = None
        #self.smiles = self.data['Smiles']
        #self.X, self.y = self.data.drop(columns = ['Smiles', 'Class']), self.data['Class']
        
       
    def _lowVarianceFS(self):
        vt = VarianceThreshold(threshold=(self.param))
        tr = vt.fit_transform(self.X)
        self.column_indexes = vt.get_support(indices = True)
        #return pd.concat([self.smiles, pd.DataFrame(vt.fit_transform(self.X)), self.y], axis= 1, sort = False) 
        return pd.DataFrame(tr)
        
    def _KbestFS(self):
        kb = SelectKBest(chi2, k=self.param)
        X_new = kb.fit_transform(self.X, self.y)
        self.column_indexes = kb.get_support(indices = True)
        #return pd.concat([self.smiles, pd.DataFrame(X_new), self.y], axis=1, sort=False)
        return pd.DataFrame(X_new)
    
    def _percentilFS(self):
        sp = SelectPercentile(chi2, percentile=self.param)
        X_new = sp.fit_transform(self.X, self.y)
        self.column_indexes = sp.get_support(indices = True)
        #return pd.concat([self.smiles, pd.DataFrame(X_new), self.y], axis=1, sort=False)
        return pd.DataFrame(X_new)
    
    def _RFECVFS(self):
        rfe = RFECV(RandomForestClassifier(n_jobs = -1), step=1, cv=5)
        X_new = rfe.fit_transform(self.X, self.y)
        self.column_indexes = rfe.get_support(indices = True)
        #return pd.concat([self.smiles, pd.DataFrame(X_new), self.y], axis=1, sort=False)
        return pd.DataFrame(X_new)
    
    def _selectFromModel(self):
        sfm= SelectFromModel(RandomForestClassifier(n_jobs=-1), threshold= self.param)
        X_new = sfm.fit_transform(self.X, self.y)
        self.column_indexes = sfm.get_support(indices = True)
        #return pd.concat([self.smiles, pd.DataFrame(X_new), self.y], axis=1, sort=False)
        return pd.DataFrame(X_new)
 
    
    def get_fsDataset(self):
        if self.fsType == 'lowVariance':
            return self._lowVarianceFS()
        elif self.fsType == 'kbest':
            return self._KbestFS()
        elif self.fsType == 'percentile':
            return self._percentilFS()
        elif self.fsType == 'RFECV':
            return self._RFECVFS()
        elif self.fsType == 'selectFromModel':
            return self._selectFromModel()
        elif self.fsType == 'keepAll':
            return self.X
        else :
            print('Invalid model type!')
            return None
        
        
if __name__ == '__main__':
    #df = pd.read_csv('merged_dataset_final2.csv', sep = ';', header = 0)
    #mrg_fps = fingerprintGenerator(df)
    #mrg_dataset = mrg_fps.getFingerprintsDataset()
    pass