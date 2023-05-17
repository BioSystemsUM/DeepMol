from abc import abstractmethod

from deepmol.base import Estimator
from deepmol.datasets import Dataset


class Transformer(Estimator):
    """
    Abstract base class for transformers.
    A transformer is an object that can transform a Dataset object.
    """

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset.
        The transformer needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        if not self.is_fitted:
            raise ValueError('Transformer needs to be fitted before calling transform()')
        return self._transform(dataset)

    @abstractmethod
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit the transformer to the dataset and transform it.
        Equivalent to calling fit(dataset) and then transform(dataset).

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
