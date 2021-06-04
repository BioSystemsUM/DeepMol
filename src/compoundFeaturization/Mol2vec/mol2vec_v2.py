import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops

from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, sentences2vec, DfVec

from typing import Any, Optional

from Datasets.Datasets import Dataset
class Mol2Vec():
    '''Mol2Vec fingerprint implementation from https://doi.org/10.1021/acs.jcim.7b00616
    Inspired by natural language processing techniques, Mol2vec, which is an unsupervised machine learning
    approach to learn vector representations of molecular substructures. Mol2vec learns vector representations
    of molecular substructures that point in similar directions for chemically related substructures.
    Compounds can finally be encoded as vectors by summing the vectors of the individual substructures and,
    for instance, be fed into supervised machine learning approaches to predict compound properties.
    '''

    def __init__(self):
        self.model = word2vec.Word2Vec.load('/content/model_300dim.pkl')
        
    
    def featurize(self, dataset: Dataset, log_every_n=1000):
        molecules = [Chem.MolFromSmiles(x) for x in dataset.mols]
        sentences = [mol2alt_sentence(x, 2) for x in molecules]
        vectors = [DfVec(x) for x in sentences2vec(sentences, self.model, unseen='UNK')]
        dataset.X = np.array([x.vec for x in vectors])
        return dataset