from typing import Literal

from optuna import Trial

from deepmol.base import Transformer
from deepmol.compound_featurization import All3DDescriptors, TwoDimensionDescriptors, MorganFingerprint, \
    AtomPairFingerprint, LayeredFingerprint, RDKFingerprint, MACCSkeysFingerprint, Mol2Vec

_1D_FEATURIZERS = {'3d_descriptors': All3DDescriptors, '2d_descriptors': TwoDimensionDescriptors,
                   'morgan': MorganFingerprint, 'atom_pair': AtomPairFingerprint, 'layered': LayeredFingerprint,
                   'rdk': RDKFingerprint, 'maccs': MACCSkeysFingerprint, 'mol2vec': Mol2Vec}

# TODO: add one-hot encoding and other featurizers from deepchem that are not graph-based + MixedFeaturizer


def _get_featurizer(trial: Trial, feat_type: Literal['1D', '2D', '3D']) -> Transformer:
    if feat_type == '1D':
        feat = trial.suggest_categorical('1D_featurizer', list(_1D_FEATURIZERS.keys()))
        if feat == 'morgan':
            radius = trial.suggest_int('radius', 2, 6, step=2)
            n_bits = trial.suggest_int('n_bits', 1024, 2048, step=1024)
            return MorganFingerprint(radius=radius, size=n_bits)
        elif feat == 'atom_pair':
            n_bits = trial.suggest_int('n_bits', 1024, 2048, step=1024)
            min_length = trial.suggest_int('min_length', 1, 3)
            max_length = trial.suggest_int('max_length', 20, 50, step=10)
            return AtomPairFingerprint(size=n_bits, min_length=min_length, max_length=max_length)
        elif feat == 'layered':
            n_bits = trial.suggest_int('n_bits', 1024, 2048, step=1024)
            min_path = trial.suggest_int('min_length', 1, 3)
            max_path = trial.suggest_int('max_length', 5, 10)
            return LayeredFingerprint(size=n_bits, min_path=min_path, max_path=max_path)
        elif feat == 'rdk':
            fpSize = trial.suggest_int('fpSize', 1024, 2048, step=1024)
            min_path = trial.suggest_int('min_path', 1, 3)
            max_path = trial.suggest_int('max_path', 5, 10)
            return RDKFingerprint(fpSize=fpSize, minPath=min_path, maxPath=max_path)
        return _1D_FEATURIZERS[feat]()
