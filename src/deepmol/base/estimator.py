from abc import abstractmethod

from deepmol.base._serializer import Serializer
from deepmol.datasets import Dataset


class Estimator(Serializer):
    """
    Abstract base class for estimators.
    An estimator is an object that can be fitted to a Dataset object.
    """

    def __init__(self):
        """
        Initialize the estimator.
        """
        self.is_fitted_ = False

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
        self.is_fitted_ = True
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

    @property
    def is_fitted(self) -> bool:
        """
        Whether the estimator is fitted.

        Returns
        -------
        is_fitted: bool
            Whether the estimator is fitted.
        """
        return hasattr(self, 'is_fitted_') and self.is_fitted_
