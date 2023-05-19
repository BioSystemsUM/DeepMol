from typing import Tuple

from sklearn import preprocessing

from deepmol.scalers.base_scaler import BaseScaler


class StandardScaler(BaseScaler):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self, copy: bool = True, with_mean: bool = True, with_std: bool = True, columns: list = None) -> None:
        """
        Constructor for the StandardScaler class.

        Parameters
        ----------
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        with_mean: bool
            If True, center the data before scaling.
        with_std: bool
            If True, scale the data to unit variance (or equivalently, unit standard deviation).
        columns: list
            The columns to be scaled.
        """
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler_object = preprocessing.StandardScaler(copy=self.copy,
                                                           with_mean=self.with_mean,
                                                           with_std=self.with_std)
        super().__init__(scaler=self._scaler_object, columns=columns)


class MinMaxScaler(BaseScaler):
    """
    Transform features by scaling each feature to a given range.
    """

    def __init__(self,
                 feature_range: Tuple[int, int] = (0, 1),
                 copy: bool = True,
                 clip: bool = False,
                 columns: list = None) -> None:
        """
        Constructor for the MinMaxScaler class.

        Parameters
        ----------
        feature_range: Tuple[int, int]
            Desired range of transformed data.
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        clip: bool
            If True, clip the values to the feature_range.
        columns: list
            The columns to be scaled.
        """
        self.copy = copy
        self.feature_range = feature_range
        self.clip = clip
        self._scaler_object = preprocessing.MinMaxScaler(copy=self.copy,
                                                         feature_range=self.feature_range,
                                                         clip=self.clip)
        super().__init__(scaler=self._scaler_object, columns=columns)


class MaxAbsScaler(BaseScaler):
    """
    Scale each feature by its maximum absolute value.
    """

    def __init__(self, copy: bool = True, columns: list = None) -> None:
        """
        Constructor for the MaxAbsScaler class.

        Parameters
        ----------
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        """
        self.copy = copy
        self._scaler_object = preprocessing.MaxAbsScaler(copy=self.copy)
        super().__init__(scaler=self._scaler_object, columns=columns)


class RobustScaler(BaseScaler):
    """
    Scale features using statistics that are robust to outliers.
    """

    def __init__(self,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 copy: bool = True,
                 unit_variance: bool = False,
                 columns: list = None) -> None:
        """
        Constructor for the RobustScaler class.

        Parameters
        ----------
        with_centering: bool
            If True, center the data before scaling.
        with_scaling: bool
            If True, scale the data to interquartile range.
        quantile_range: Tuple[float, float]
            Percentiles used to calculate the quantiles used for scaling.
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        unit_variance: bool
            If True, scale the data to unit variance.
        columns: list
            The columns to be scaled.
        """
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
        super().__init__(scaler=self._scaler_object, columns=columns)


class PolynomialFeatures(BaseScaler):
    """
    Generate polynomial and interaction features.
    """

    def __init__(self,
                 degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 order: str = 'C',
                 columns: list = None) -> None:
        """
        Constructor for the PolynomialFeatures class.

        Parameters
        ----------
        degree: int
            The degree of the polynomial features.
        interaction_only: bool
            If True, only interaction features are produced.
        include_bias: bool
            If True, then include a bias column, the feature in which all polynomial powers are zero.
        order: str
            C or F. Order of output array in the dense case. 'F' order is faster to compute, but may slow down
            subsequent estimators.
        columns: list
            The columns to be scaled.
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self._scaler_object = preprocessing.PolynomialFeatures(degree=self.degree,
                                                               interaction_only=self.interaction_only,
                                                               include_bias=self.include_bias,
                                                               order=self.order)
        super().__init__(scaler=self._scaler_object, columns=columns)


class Normalizer(BaseScaler):
    """
    Normalize samples individually to unit norm.
    """

    def __init__(self, norm: str = 'l2', copy: bool = True, columns: list = None) -> None:
        """
        Constructor for the Normalizer class.

        Parameters
        ----------
        norm: str
            One of 'l1', 'l2' or 'max'. The norm to use to normalize each non-zero sample.
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        """
        self.norm = norm
        self.copy = copy
        self._scaler_object = preprocessing.Normalizer(norm=self.norm, copy=self.copy)
        super().__init__(scaler=self._scaler_object, columns=columns)


class Binarizer(BaseScaler):
    """
    Binarize data (set feature values to 0 or 1) according to a threshold.
    """

    def __init__(self, threshold: float = 0.0, copy: bool = True, columns: list = None) -> None:
        """
        Constructor for the Binarizer class.

        Parameters
        ----------
        threshold: float
            The threshold parameter.
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        """
        self.threshold = threshold
        self.copy = copy
        self._scaler_object = preprocessing.Binarizer(threshold=self.threshold, copy=self.copy)
        super().__init__(scaler=self._scaler_object, columns=columns)


class KernelCenterer(BaseScaler):
    """
    Center a kernel matrix.
    """

    def __init__(self, columns: list = None) -> None:
        """
        Constructor for the KernelCenterer class.

        Parameters
        ----------
        columns: list
            The columns to be scaled.
        """
        self._scaler_object = preprocessing.KernelCenterer()
        super().__init__(scaler=self._scaler_object, columns=columns)


class QuantileTransformer(BaseScaler):
    """
    Transform features using quantiles information.
    """

    def __init__(self,
                 n_quantiles: int = 1000,
                 output_distribution: str = 'uniform',
                 ignore_implicit_zeros: bool = False,
                 subsample: int = int(1e5),
                 random_state: int = None,
                 copy: bool = True,
                 columns: list = None) -> None:
        """
        Constructor for the QuantileTransformer class.

        Parameters
        ----------
        n_quantiles: int
            Number of quantiles to be computed. It corresponds to the number of landmarks used to discretize the
            cumulative distribution function. It has to be strictly greater than the number of samples.
        output_distribution: str
            Marginal distribution for the transformed data. The choices are ‘uniform’ (default) or ‘normal’.
        ignore_implicit_zeros: bool
            Only applies to sparse matrices. If True, the sparse entries of the matrix are discarded to compute the
            quantile statistics. If false, these entries are accounting for zeros.
        subsample: int
            Maximum number of samples used to estimate the quantiles for computational efficiency.
        random_state: int
            Pseudo-random number generator state used for random sampling.
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        columns: list
            The columns to be scaled.
        """
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self._scaler_object = preprocessing.QuantileTransformer(n_quantiles=self.n_quantiles,
                                                                output_distribution=self.output_distribution,
                                                                ignore_implicit_zeros=self.ignore_implicit_zeros,
                                                                subsample=self.subsample,
                                                                random_state=self.random_state, copy=self.copy)
        super().__init__(scaler=self._scaler_object, columns=columns)


class PowerTransformer(BaseScaler):
    """
    Apply a power transform featurewise to make data more Gaussian-like.
    """

    def __init__(self,
                 method: str = 'yeo-johnson',
                 standardize: bool = True,
                 copy: bool = True,
                 columns: list = None) -> None:
        """
        Constructor for the PowerTransformer class.

        Parameters
        ----------
        method: str
            The power transform method. Available methods are:
                - 'yeo-johnson', works with positive and negative values
                - 'box-cox', only works with strictly positive values
        standardize: bool
            Set to True to apply zero mean, unit variance normalization to the transformed output.
        copy: bool
            If False, try to avoid a copy and do inplace scaling instead.
        columns: list
            The columns to be scaled.
        """
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self._scaler_object = preprocessing.PowerTransformer(copy=self.copy)
        super().__init__(scaler=self._scaler_object, columns=columns)
