from abc import abstractmethod

from deepmol.base._serializer import Serializer
from deepmol.datasets import Dataset


class Estimator(Serializer):
    """
    Abstract base class for estimators.
    An estimator is an object that can be fitted to a Dataset object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the estimator.
        """
        self._is_fitted = False

    def fit(self, dataset: Dataset) -> 'Estimator':
        """
        Fit the estimator to the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the estimator to.

        Returns
        -------
        self: Estimator
            The fitted estimator.
        """
        self._fit(dataset)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit(self, dataset: Dataset) -> 'Estimator':
        """
        Fit the estimator to the data.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the estimator to.

        Returns
        -------
        self: Estimator
            The fitted estimator.
        """

    def is_fitted(self) -> bool:
        """
        Whether the estimator is fitted.

        Returns
        -------
        is_fitted: bool
            Whether the estimator is fitted.
        """
        return hasattr(self, '_is_fitted') and self._is_fitted
