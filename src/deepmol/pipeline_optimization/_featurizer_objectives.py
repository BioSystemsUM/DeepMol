from typing import Literal

from optuna import Trial

from deepmol.base import Transformer
from deepmol.compound_featurization import All3DDescriptors, TwoDimensionDescriptors, MorganFingerprint, \
    AtomPairFingerprint, LayeredFingerprint, RDKFingerprint, MACCSkeysFingerprint, Mol2Vec, SmilesOneHotEncoder

# _1D_FEATURIZERS = {'3d_descriptors': All3DDescriptors, '2d_descriptors': TwoDimensionDescriptors,
#                    'morgan': MorganFingerprint, 'atom_pair': AtomPairFingerprint, 'layered': LayeredFingerprint,
#                    'rdk': RDKFingerprint, 'maccs': MACCSkeysFingerprint, 'mol2vec': Mol2Vec}
_1D_FEATURIZERS = {'2d_descriptors': TwoDimensionDescriptors,
                   'morgan': MorganFingerprint, 'atom_pair': AtomPairFingerprint, 'layered': LayeredFingerprint,
                   'rdk': RDKFingerprint, 'maccs': MACCSkeysFingerprint, 'mol2vec': Mol2Vec}

# TODO: add one-hot encoding and other featurizers from deepchem that are not graph-based + MixedFeaturizer


def _get_featurizer(trial: Trial, feat_type: Literal['1D', '2D']) -> Transformer:
    if feat_type == '1D':
        feat = trial.suggest_categorical('1D_featurizer', list(_1D_FEATURIZERS.keys()))
        if feat == 'morgan':
            radius = trial.suggest_int('radius', 2, 6, step=2)
            n_bits = trial.suggest_int('n_bits', 1024, 2048, step=1024)
            return MorganFingerprint(radius=radius, size=n_bits)
        elif feat == 'atom_pair':
            nBits = trial.suggest_int('nBits', 1024, 2048, step=1024)
            minLength = trial.suggest_int('minLength', 1, 3)
            maxLength = trial.suggest_int('maxLength', 20, 50, step=10)
            return AtomPairFingerprint(nBits=nBits, minLength=minLength, maxLength=maxLength)
        elif feat == 'layered':
            fpSize = trial.suggest_int('fpSize', 1024, 2048, step=1024)
            minPath = trial.suggest_int('minPath', 1, 3)
            maxPath = trial.suggest_int('maxPath', 5, 10)
            return LayeredFingerprint(fpSize=fpSize, minPath=minPath, maxPath=maxPath)
        elif feat == 'rdk':
            fpSize = trial.suggest_int('fpSize', 1024, 2048, step=1024)
            min_path = trial.suggest_int('min_path', 1, 3)
            max_path = trial.suggest_int('max_path', 5, 10)
            return RDKFingerprint(fpSize=fpSize, minPath=min_path, maxPath=max_path)
        return _1D_FEATURIZERS[feat]()
    elif feat_type == '2D':
        return SmilesOneHotEncoder()
    else:
        raise ValueError(f'Unknown featurizer type {feat_type}')
