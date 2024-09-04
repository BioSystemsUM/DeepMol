import os
from typing import Iterable

import numpy as np
from gensim.models import Word2Vec, word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
from rdkit.Chem import Mol

from deepmol.compound_featurization import MolecularFeaturizer


def sentences2vec(sentences: Iterable, model: Word2Vec, unseen: str = None):
    """
    Generate vectors for each sentence (list) in a list of sentences. Vector is simply a sum of vectors for individual
    words.

    Parameters
    ----------
    sentences : Iterable
        List with sentences
    model : Word2Vec
        Gensim Word2Vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
        Array of vectors for each sentence.
    """

    keys = set(model.wv.key_to_index)
    vec = []

    for sentence in sentences:
        if unseen:
            unseen_vec = model.wv.get_vector(unseen)
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))
    return np.array(vec)


class Mol2Vec(MolecularFeaturizer):
    """
    Mol2Vec fingerprint implementation from https://doi.org/10.1021/acs.jcim.7b00616

    Inspired by natural language processing techniques, Mol2vec, which is an unsupervised machine learning
    approach to learn vector representations of molecular substructures. Mol2vec learns vector representations
    of molecular substructures that point in similar directions for chemically related substructures.
    Compounds can finally be encoded as vectors by summing the vectors of the individual substructures and,
    for instance, be fed into supervised machine learning approaches to predict compound properties.
    """

    def __init__(self, pretrain_model_path: str = None,
                 radius: int = 1,
                 unseen: str = 'UNK',
                 gather_method: str = 'sum', **kwargs):

        """
        Parameters
        ----------
        pretrain_model_path: str
            Path to pretrained model. If this value is None, we use the model_300dim.pkl model.
            The model is trained on 20 million compounds downloaded from ZINC.
        radius: int
            The fingerprint radius. The default value was used to train the model_300dim.pkl model.
        unseen: str
            The string to used to replace uncommon words/identifiers while training.
        gather_method: str
            How to aggregate vectors of identifiers are extracted from Mol2vec. 'sum' or 'mean' is supported.
        """

        super().__init__(**kwargs)
        self.radius = radius
        self.unseen = unseen
        self.gather_method = gather_method
        self.sentences2vec = sentences2vec
        if pretrain_model_path is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pretrain_model_path = os.path.join(BASE_DIR,
                                               "compound_featurization",
                                               "mol2vec_models",
                                               "model_300dim.pkl")
        self.model = word2vec.Word2Vec.load(pretrain_model_path)
        self.feature_names = [f"mol2vec_{i}" for i in range(self.model.vector_size)]

    def _featurize(self, mol: Mol):
        """
        Calculate mol2vec fingerprints.
        Parameters
        ----------
        mol: Mol
          RDKit Mol object
        Returns
        -------
        features: np.ndarray
          1D array of mol2vec fingerprint. The default length is 300.
        """
        # try:
        sentence = MolSentence(mol2alt_sentence(mol, self.radius))
        vec_identifiers = self.sentences2vec(
            sentence, self.model, unseen=self.unseen)
        if self.gather_method == 'sum':
            feature = np.sum(vec_identifiers, axis=0)
        elif self.gather_method == 'mean':
            feature = np.mean(vec_identifiers, axis=0)
        else:
            raise ValueError(
                'Not supported gather_method type. Please set "sum" or "mean"')
        return feature
