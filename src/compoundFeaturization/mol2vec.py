import numpy as np
from typing import Any, Optional
from compoundFeaturization.baseFeaturizer import MolecularFeaturizer
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, sentences2vec
import os, sys

class Mol2Vec(MolecularFeaturizer):
    '''Mol2Vec fingerprint implementation from https://doi.org/10.1021/acs.jcim.7b00616

    Inspired by natural language processing techniques, Mol2vec, which is an unsupervised machine learning
    approach to learn vector representations of molecular substructures. Mol2vec learns vector representations
    of molecular substructures that point in similar directions for chemically related substructures.
    Compounds can finally be encoded as vectors by summing the vectors of the individual substructures and,
    for instance, be fed into supervised machine learning approaches to predict compound properties.
    '''

    def __init__(self,
                 pretrain_model_path: Optional[str] = None,
                 radius: int = 1,
                 unseen: str = 'UNK',
                 gather_method: str = 'sum'):

        '''
        Parameters
        ----------
        pretrain_file: str, optional
            The path for pretrained model. If this value is None, we use the model_300dim.pkl model.
            The model is trained on 20 million compounds downloaded from ZINC.
        radius: int, optional (default 1)
            The fingerprint radius. The default value was used to train the model_300dim.pkl model.
        unseen: str, optional (default 'UNK')
            The string to used to replace uncommon words/identifiers while training.
        gather_method: str, optional (default 'sum')
            How to aggregate vectors of identifiers are extracted from Mol2vec. 'sum' or 'mean' is supported.
        '''

        self.radius = radius
        self.unseen = unseen
        self.gather_method = gather_method
        self.sentences2vec = sentences2vec
        self.mol2alt_sentence = mol2alt_sentence
        if pretrain_model_path is None:
            pretrain_model_path = 'compoundFeaturization/mol2vec_models/model_300dim.pkl'
        self.model = word2vec.Word2Vec.load(pretrain_model_path)

    def _featurize(self, mol: Any) -> np.ndarray:
        """
        Calculate mol2vec fingerprints.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          1D array of mol2vec fingerprint. The default length is 300.
        """
        try :
            sentence = self.mol2alt_sentence(mol, self.radius)
            vec_identifiers = self.sentences2vec(
                sentence, self.model, unseen=self.unseen)
            if self.gather_method == 'sum':
                feature = np.sum(vec_identifiers, axis=0)
            elif self.gather_method == 'mean':
                feature = np.mean(vec_identifiers, axis=0)
            else:
                raise ValueError(
                    'Not supported gather_method type. Please set "sum" or "mean"')
        except Exception as e:
            feature = np.asarray(np.nan)
        return feature