# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:16:54 2019

@author: jfsco
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, DfVec, sentences2vec
import deepchem as dc
from deepchem.models.tensorgraph.optimizers import Adam, ExponentialDecay

class embeddingsGenerator():
    
    def __init__(self, dataset, smiles_label, class_label = None, emb_type = 'mol2vec', model = None):
        #self.dataset = dataset[['Smiles','Class']]
        if class_label is not None:
            self.dataset = pd.DataFrame({'Smiles': dataset[smiles_label].tolist(), 'Class': dataset[class_label].tolist()})
            self.labeled = True
        else :
            self.dataset = pd.DataFrame({'Smiles': dataset[smiles_label].tolist()})
            self.labeled = False
        self.emb_type = emb_type
        self.dataset['Molecules'] = [Chem.MolFromSmiles(x) for x in self.dataset.Smiles]
        # dropna will change the index of the smiles if, for instance, some of the smiles are not valid and the embedding cannot be computed
        # keep it that way or let the smiles be there with NAs as features to maintain the order?
        self.dataset = self.dataset.dropna()
        self.smiles_list = self.dataset.Smiles.tolist()
        if self.labeled:
            self.labels_list = self.dataset.Class.tolist()
        if emb_type == 'mol2vec':
            self.model = word2vec.Word2Vec.load(model)
        
    
    def _mol2vec_emb(self):
        sentences = [mol2alt_sentence(x, 2) for x in self.dataset.Molecules]
        vectors = [DfVec(x) for x in sentences2vec(sentences, self.model, unseen='UNK')]
        vec_df = pd.DataFrame(data=np.array([x.vec for x in vectors]))
        vec_df.columns = ['mol2vec_' + str(x+1) for x in vec_df.columns.values]
        vec_df.index = self.dataset.index.values
        self.dataset = self.dataset.drop(['Molecules'], axis = 1)
        return pd.concat([self.dataset, vec_df], axis=1)
    
    
    def _getAlphabet(self):
        tokens = set()
        for sm in self.smiles_list:
            tokens = tokens.union(set(s for s in sm))
        return sorted(list(tokens))
    
    
    def _generateSequences(self, epochs):
        for i in range(epochs):
            for s in self.smiles_list:
                yield (s, s)
    
    
    def _seq2seqModel(self):
        
        max_len = max(len(s) for s in self.smiles_list)
        tokens = self._getAlphabet()
        model = dc.models.SeqToSeq(tokens,
                           tokens,
                           max_len,
                           encoder_layers=2,
                           decoder_layers=2,
                           embedding_dimension=256,
                           model_dir='fingerprint')
        batches_per_epoch = len(self.smiles_list)/model.batch_size
        model.set_optimizer(Adam(learning_rate=ExponentialDecay(0.004, 0.9, batches_per_epoch)))
        
        model.fit_sequences(self._generateSequences(40))
        
        embeddings = model.predict_embeddings(self.smiles_list)
        if self.labeled:
            embeddings_dataset = dc.data.NumpyDataset(embeddings,
                                                      self.smiles_list,
                                                      self.labels_list)
        else:
            embeddings_dataset = dc.data.NumpyDataset(embeddings,
                                                      self.smiles_list)
        return embeddings_dataset
    
    
    def _seq2seq_emb(self):
        model = self._seq2seqModel()
        df = pd.DataFrame(data=model.X)
        df.columns = ['seq2seq_' + str(x+1) for x in df.columns.values]
        df.insert(loc = 0, column = 'Smiles', value = model.y)
        if self.labeled:
            df.insert(loc = 1, column = 'Class', value = self.labels_list)
            return df
        else :
            return df
        
    
    def getEmbeddingsDataset(self):

        if self.emb_type == 'mol2vec':
            return self._mol2vec_emb()
        elif self.emb_type == 'seq2seq':
            return self._seq2seq_emb()
        else :
            print('Invalid embedding name!')
            return pd.DataFrame()
        
        
if __name__ == '__main__':
    df = pd.read_csv('dataset_last_version2.csv', sep = ';', header = 0)[:10]
    embeddings = embeddingsGenerator(df, 'Smiles', 'Class', 'seq2seq')#, 'mol2vec_models/model_300dim.pkl')
    seq2seq_df = embeddings.getEmbeddingsDataset()
    #print(mol2vec_df.columns)