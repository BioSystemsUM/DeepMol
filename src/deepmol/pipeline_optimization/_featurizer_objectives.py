import os
from typing import Literal

from optuna import Trial

from deepmol.base import Transformer
from deepmol.compound_featurization import *

try:
    _1D_FEATURIZERS = {'2d_descriptors': TwoDimensionDescriptors, 'morgan': MorganFingerprint,
                       'atom_pair': AtomPairFingerprint, 'layered': LayeredFingerprint, 'rdk': RDKFingerprint,
                       'maccs': MACCSkeysFingerprint, 'mol2vec': Mol2Vec, 'mixed': MixedFeaturizer,
                       'np_classifier_fp': NPClassifierFP, 'neural_npfp': NeuralNPFP, "mhfp": MHFP}
except NameError:
    warnings.warn("Mol2Vec featurizer not available. If you want to use it install it, please.")
    _1D_FEATURIZERS = {'2d_descriptors': TwoDimensionDescriptors, 'morgan': MorganFingerprint,
                       'atom_pair': AtomPairFingerprint, 'layered': LayeredFingerprint, 'rdk': RDKFingerprint,
                       'maccs': MACCSkeysFingerprint, 'mixed': MixedFeaturizer, 'np_classifier_fp': NPClassifierFP,
                       'neural_npfp': NeuralNPFP, "mhfp": MHFP}


def _get_featurizer(trial: Trial, feat_type: Literal['1D', '2D']) -> Transformer:
    """
    Optuna objective function for featurizers.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    feat_type : Literal['1D', '2D']
        The type of the featurizer. Either '1D' or '2D'.

    Returns
    -------
    Transformer
        The featurizer.
    """
    num_of_cores = os.cpu_count() // 2
    if feat_type == '1D':
        feat = trial.suggest_categorical('1D_featurizer', list(_1D_FEATURIZERS.keys()))
        if feat == 'morgan':
            radius = trial.suggest_int('radius', 2, 6, step=2)
            n_bits = trial.suggest_int('n_bits', 1024, 2048, step=1024)
            return MorganFingerprint(radius=radius, size=n_bits, n_jobs=num_of_cores)
        elif feat == 'atom_pair':
            nBits = trial.suggest_int('nBits', 1024, 2048, step=1024)
            minLength = trial.suggest_int('minLength', 1, 3)
            maxLength = trial.suggest_int('maxLength', 20, 50, step=10)
            return AtomPairFingerprint(nBits=nBits, minLength=minLength, maxLength=maxLength, n_jobs=num_of_cores)
        elif feat == 'layered':
            fpSize = trial.suggest_int('fpSize', 1024, 2048, step=1024)
            minPath = trial.suggest_int('minPath', 1, 3)
            maxPath = trial.suggest_int('maxPath', 5, 10)
            return LayeredFingerprint(fpSize=fpSize, minPath=minPath, maxPath=maxPath, n_jobs=num_of_cores)
        elif feat == 'rdk':
            fpSize = trial.suggest_int('fpSize', 1024, 2048, step=1024)
            min_path = trial.suggest_int('min_path', 1, 3)
            max_path = trial.suggest_int('max_path', 5, 10)
            return RDKFingerprint(fpSize=fpSize, minPath=min_path, maxPath=max_path, n_jobs=num_of_cores)
        elif feat == 'mixed':
            available_feats = list(_1D_FEATURIZERS.keys())
            available_feats.remove('mixed')
            f1 = trial.suggest_categorical('f1', available_feats)
            f2 = trial.suggest_categorical('f2', available_feats)
            if f1 == f2:
                while f1 == f2:
                    f2 = trial.suggest_categorical('f2', available_feats)
            return MixedFeaturizer([_1D_FEATURIZERS[f1](), _1D_FEATURIZERS[f2]()], n_jobs=num_of_cores)
        elif feat == 'np_classifier_fp':
            return NPClassifierFP(n_jobs=num_of_cores)
        elif feat == 'neural_npfp':
            model = trial.suggest_categorical('model', ['aux', 'base', 'ae'])
            return NeuralNPFP(model, n_jobs=num_of_cores)
        elif feat == 'mhfp':
            return MHFP(n_jobs=num_of_cores)

        return _1D_FEATURIZERS[feat](n_jobs=num_of_cores)
    elif feat_type == '2D':
        return SmilesOneHotEncoder(n_jobs=num_of_cores)
    else:
        raise ValueError(f'Unknown featurizer type {feat_type}')
