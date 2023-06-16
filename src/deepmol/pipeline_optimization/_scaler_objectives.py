from deepmol.base import PassThroughTransformer, Transformer
from deepmol.scalers import StandardScaler, MinMaxScaler, MaxAbsScaler, Binarizer, Normalizer, RobustScaler, \
    KernelCenterer, QuantileTransformer, PowerTransformer

# TODO: PolynomialFeatures only woks with square matrices
# TODO: KernelCenter only woks with square matrices
_SCALERS = {'standard_scaler': StandardScaler, 'min_max_scaler': MinMaxScaler, 'max_abs_scaler': MaxAbsScaler,
            'robust_scaler': RobustScaler, 'normalizer': Normalizer, 'binarizer': Binarizer,
            'quantile_transformer': QuantileTransformer,
            'power_transformer': PowerTransformer, 'pass_through_transformer': PassThroughTransformer}


def _get_scaler(trial) -> Transformer:
    scaler = trial.suggest_categorical("scaler", list(_SCALERS.keys()))
    # TODO: add parameters for some scalers
    return _SCALERS[scaler]()
