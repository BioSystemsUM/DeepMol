from abc import ABC
from typing import Union, Iterable

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, RFECV, SelectFromModel, \
    f_classif

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.utils.decorators import modify_object_inplace_decorator


class BaseFeatureSelector(ABC, Transformer):
    """
    Abstract class for feature selection.
    A `BaseFeatureSelector` uses features present in a Dataset object to select the most important ones.
    FeatureSelectors which are subclasses of this class should always operate over Dataset Objects.
    """

    def __init__(self, feature_selector):
        """
        Initialize the feature selector.
        """
        if self.__class__ == BaseFeatureSelector:
            raise Exception('Abstract class BaseFeatureSelector should not be instantiated')
        super().__init__()
        self.feature_selector = feature_selector
        self.features_to_keep = None

    @modify_object_inplace_decorator
    def select_features(self, dataset: Dataset) -> Dataset:
        """
        Perform feature selection for the molecules present in the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to perform feature selection on
        inplace: bool
            Whether to perform the feature selection in the received dataset or not.

        Returns
        -------
        dataset: Dataset
          Dataset containing the selected features and indexes of the features kept as 'self.features2keep'.
        """
        return self.fit_transform(dataset)

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Perform feature selection for the molecules present in the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to perform feature selection on

        Returns
        -------
        dataset: Dataset
          Dataset containing the selected features and indexes of the features kept as 'self.features2keep'.
        """
        if self.features_to_keep is not None:
            dataset = dataset.select_features_by_index(list(self.features_to_keep))
        return dataset

    def _fit(self, dataset: Dataset) -> 'BaseFeatureSelector':
        """
        Fits the feature selector to a dataset of molecules.

        Parameters
        ----------
        dataset: Dataset
            Dataset of molecules.

        Returns
        -------
        self: BaseFeatureSelector
            The fitted feature selector.
        """
        x = np.stack(dataset.X, axis=0)
        y = np.array(dataset.y)
        fs = self.feature_selector.fit(x, y)
        self.features_to_keep = fs.get_support(indices=True)
        return self


class LowVarianceFS(BaseFeatureSelector):
    """
    Class for Low Variance feature selection.
    Feature selector that removes all features with low-variance.
    """

    def __init__(self, threshold: float = 0.3):
        """
        Initialize this Feature Selector

        Parameters
        ----------
        threshold: float
            Features with a training-set variance lower than this threshold will be removed.
        """
        self.param = threshold
        super().__init__(VarianceThreshold(threshold=threshold))


class KbestFS(BaseFeatureSelector):
    """
    Class for K best feature selection.

    Select features according to the k-highest scores.
    """

    def __init__(self, k: int = 10, score_func: callable = f_classif):
        """
        Initialize this KbestFS Feature Selector.

        Parameters
        ----------
        k: int
            Number of top features to select.
        score_func: callable
            Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with
            scores.
        """
        super().__init__(SelectKBest(score_func=score_func, k=k))


class PercentilFS(BaseFeatureSelector):
    """
    Class for percentil feature selection.

    Select features according to a percentile of the highest scores.
    """

    def __init__(self, percentil: int = 10, score_func: callable = f_classif):
        """
        Initialize the PercentilFS Feature Selector.

        Parameters
        ----------
        percentil: int
            Percent of features to keep.
        score_func: callable
            Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with
            scores.
        """
        super().__init__(SelectPercentile(score_func=score_func, percentile=percentil))


# TODO: takes too long to run, check if its normal or a code problem
class RFECVFS(BaseFeatureSelector):
    """
    Class for RFECV feature selection.

    Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
    """

    def __init__(self,
                 estimator: callable = None,
                 step: Union[int, float] = 1,
                 min_features_to_select: int = 1,
                 cv: Union[int, callable, Iterable] = None,
                 scoring: Union[str, callable] = None,
                 verbose: int = 0,
                 n_jobs: int = -1):
        """
        Initialize the RFECVFS Feature Selector.

        Parameters
        ----------
        estimator: callable
            A supervised learning estimator with a fit method that provides information about feature importance either
            through a coef_ attribute or through a feature_importances_ attribute.
        step: Union[int, float]
            If greater than or equal to 1, then step corresponds to the (integer) number of features to remove
            at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of
            features to remove at each iteration. Note that the last iteration may remove fewer than step features
            in order to reach min_features_to_select.
        min_features_to_select: int
            The minimum number of features to be selected. This number of features will always be scored, even if
            the difference between the original feature count and min_features_to_select isn’t divisible by step.
        cv: Union[int, callable, Iterable]
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                - None, to use the default 5-fold cross-validation,
                - integer, to specify the number of folds.
                - CV splitter,
                - An iterable yielding (train, test) splits as arrays of indices.
        scoring: Union[str, callable]
            A string (see model evaluation documentation) or a scorer callable object / function with signature
            scorer(estimator, X, y).
        verbose: in
            Controls verbosity of output.
        n_jobs: int
            Number of cores to run in parallel while fitting across folds. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_jobs=n_jobs)

        rfe = RFECV(estimator=estimator,
                    step=step,
                    cv=cv,
                    min_features_to_select=min_features_to_select,
                    scoring=scoring,
                    verbose=verbose)
        super().__init__(rfe)


class SelectFromModelFS(BaseFeatureSelector):
    """
    Class for Select From Model feature selection.

    Meta-transformer for selecting features based on importance weights.
    """

    def __init__(self,
                 estimator: callable = None,
                 threshold: Union[str, float] = None,
                 prefit: bool = False,
                 norm_order: int = 1,
                 max_features: int = None):
        """
        Initialize this SelectFromModelFS Feature Selector.

        Parameters
        ----------
        estimator: callable
            The base estimator from which the transformer is built. This can be both a fitted (if prefit is set to True)
            or a non-fitted estimator. The estimator must have either a feature_importances_ or coef_ attribute after
            fitting.
        threshold: Union[str, float]
            The threshold value to use for feature selection. Features whose importance is greater or equal
            are kept while the others are discarded. If “median” (resp. “mean”), then the threshold value is the
            median (resp. the mean) of the feature importances. A scaling factor (e.g., “1.25*mean”) may also be used.
            If None and if the estimator has a parameter penalty set to l1, either explicitly or implicitly
            (e.g, Lasso), the threshold used is 1e-5. Otherwise, “mean” is used by default.
        prefit: bool
            Whether a prefit model is expected to be passed into the constructor directly or not. If True,
            transform must be called directly and SelectFromModel cannot be used with cross_val_score, GridSearchCV
            and similar utilities that clone the estimator. Otherwise, train the model using fit and then transform
            to do feature selection.
        norm_order: int
            Order of the norm used to filter the vectors of coefficients below threshold in the case where the
            coef_ attribute of the estimator is of dimension 2.
        max_features: int
            The maximum number of features to select. To only select based on max_features, set threshold=-np.inf
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_jobs=-1)
        sfm = SelectFromModel(estimator=estimator,
                              threshold=threshold,
                              prefit=prefit,
                              norm_order=norm_order,
                              max_features=max_features)
        super().__init__(sfm)


class BorutaAlgorithm(BaseFeatureSelector):
    """
    Class for Boruta feature selection.

    Boruta is an all-relevant feature selection method. It is based on the idea that all features are relevant until
    proven irrelevant. The algorithm is an iterative procedure that consists of two phases: the first phase randomly
    permutes the feature values and evaluates the performance of the classifier. The second phase eliminates the
    features that are less important than their shadow features. The shadow features are copies of the original
    features that are randomly permuted. The algorithm stops when all features are either declared important or
    declared irrelevant.
    """

    def __init__(self,
                 estimator: callable = None,
                 task: str = "classification",
                 support_weak: bool = False,
                 n_estimators: Union[int, str] = 1000,
                 perc: int = 100,
                 alpha: float = 0.05,
                 two_step: bool = True,
                 max_iter: int = 100,
                 random_state: int = None,
                 verbose: int = 0):
        """
        Initialize this BorutaAlgorithm Feature Selector.

        Parameters
        ----------
        estimator: callable
            A supervised learning estimator, with a 'fit' method that returns the feature_importances_ attribute.
            Important features must correspond to high absolute values in the feature_importances_.
        task: str
            The task to perform. Either "classification" or "regression".
        support_weak: bool
            Whether to support weak features or not. If True, weak features are also selected.
        n_estimators: Union[int, str]
            If int sets the number of estimators in the chosen ensemble method. If 'auto' this is determined
            automatically based on the size of the dataset. The other parameters of the used estimators need to be set
            with initialisation.
        perc: int
            Instead of the max we use the percentile defined by the user, to pick our threshold for comparison between
            shadow and real features. The max tend to be too stringent. This provides a finer control over this. The
            lower perc is the more false positives will be picked as relevant but also the less relevant features will
            be left out. The usual trade-off. The default is essentially the vanilla Boruta corresponding to the max.
        alpha: float
             Level at which the corrected p-values will get rejected in both correction steps.
        two_step: bool
            If you want to use the original implementation of Boruta with Bonferroni correction only set this to False.
        max_iter: int
            Maximum number of iterations to perform.
        random_state: int
            Random state to use.
        verbose: int
            Controls verbosity of output.
            - 0: no output
            - 1: displays iteration number
            - 2: which features have been selected already
        """
        self.support_weak = support_weak
        if estimator is None:
            if task == "classification":
                estimator = RandomForestClassifier(
                    n_jobs=-1,
                    max_depth=5
                )
            elif task == "regression":
                estimator = RandomForestRegressor(
                    n_jobs=-1,
                    max_depth=5
                )

        boruta = BorutaPy(
            estimator,
            n_estimators,
            perc,
            alpha,
            two_step,
            max_iter,
            random_state,
            verbose
        )
        super().__init__(boruta)

    def _fit(self, dataset: Dataset) -> 'BorutaAlgorithm':
        """
        Fit the Boruta Algorithm.

        Parameters
        ----------
        dataset: Dataset
            Dataset to fit

        Returns
        -------
        self: BorutaAlgorithm
            The fitted BorutaAlgorithm
        """
        fs = np.stack(dataset.X, axis=0)
        y = dataset.y
        self.feature_selector.fit(fs, y)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset using the selected features.

        Parameters
        ----------
        dataset: Dataset
            Dataset to transform

        Returns
        -------
        transformed_dataset: Dataset
            Transformed dataset
        """
        fs = np.stack(dataset.X, axis=0)
        self.feature_selector.transform(fs, weak=self.support_weak)
        support = [i for i, boolean in enumerate(self.feature_selector.support_) if boolean]
        if self.support_weak:
            weak_support = [i for i, boolean in enumerate(self.feature_selector.support_weak_) if boolean]
            features_to_keep = list(set.union(set(support), set(weak_support)))
        else:
            features_to_keep = support
        dataset.select_features_by_index(list(features_to_keep))
        return dataset
