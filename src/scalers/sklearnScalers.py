import joblib
import numpy as np
import sklearn.preprocessing as preprocessing

from scalers.baseScaler import BaseScaler


class StandardScaler(BaseScaler):

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler_object = preprocessing.StandardScaler(copy=self.copy,
                                                           with_mean=self.with_mean,
                                                           with_std=self.with_std)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class MinMaxScaler(BaseScaler):

    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        self.copy = copy
        self.feature_range = feature_range
        self.clip = clip
        self._scaler_object = preprocessing.MinMaxScaler(copy=self.copy,
                                                                 feature_range=self.feature_range,
                                                                 clip=self.clip)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class MaxAbsScaler(BaseScaler):

    def __init__(self, copy=True):
        self.copy = copy
        self._scaler_object = preprocessing.MaxAbsScaler(copy=self.copy)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class RobustScaler(BaseScaler):

    def __init__(self, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
        self.copy = copy
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self._scaler_object = preprocessing.RobustScaler(with_centering=self.with_centering,
                                                                 with_scaling=self.with_scaling,
                                                                 quantile_range=self.quantile_range,
                                                                 copy=self.copy,
                                                                 unit_variance=self.unit_variance)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class PolynomialFeatures(BaseScaler):

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order='C'):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self._scaler_object = preprocessing.PolynomialFeatures(degree=self.degree,
                                                                       interaction_only=self.interaction_only,
                                                                       include_bias=self.include_bias,
                                                                       order=self.order)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class Normalizer(BaseScaler):

    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy
        self._scaler_object = preprocessing.Normalizer(norm=self.norm, copy=self.copy)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class Binarizer(BaseScaler):

    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy
        self._scaler_object = preprocessing.Binarizer(threshold=self.threshold, copy=self.copy)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class KernelCenterer(BaseScaler):

    def __init__(self):
        self._scaler_object = preprocessing.KernelCenterer()

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class QuantileTransformer(BaseScaler):

    def __init__(self, n_quantiles=1000, output_distribution='uniform',
                 ignore_implicit_zeros=False, subsample=int(1e5),
                 random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self._scaler_object = \
            preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,
                                                      output_distribution=self.output_distribution,
                                                      ignore_implicit_zeros=self.ignore_implicit_zeros,
                                                      subsample=self.subsample,
                                                      random_state=self.random_state, copy=self.copy)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)


class PowerTransformer(BaseScaler):

    def __init__(self, method='yeo-johnson', standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self._scaler_object = \
            preprocessing.QuantileTransformer(method=self.method,
                                                      standardize=self.standardize,
                                                      copy=self.copy)

        super().__init__()

    @property
    def scaler_object(self):
        return self._scaler_object

    @scaler_object.setter
    def scaler_object(self, value):
        self._scaler_object = value

    def load_scaler(self, file_path):
        self.scaler_object = joblib.load(file_path)

    def _fit_transform(self, X):
        return self.scaler_object.fit_transform(X)

    def _fit(self, X: np.ndarray):
        return self.scaler_object.fit(X)

    def _transform(self, X: np.ndarray):
        return self.scaler_object.transform(X)
