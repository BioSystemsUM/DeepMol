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


class PassThroughTransformer(Transformer):
    """
    A transformer that does nothing.
    """

    def _fit(self, dataset: Dataset) -> 'PassThroughTransformer':
        """
        Fit the transformer to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the transformer to.

        Returns
        -------
        self: Estimator
            The fitted transformer.
        """
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        return dataset


class DatasetTransformer(Transformer):
    """
    A transformer that transforms a dataset by applying a function to it.
    """

    def __init__(self, func, **kwargs):
        """
        Parameters
        ----------
        func: callable
            The function to apply to the dataset.
        kwargs: dict
            Additional keyword arguments to pass to the function.
        """
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        return self.func(dataset, **self.kwargs)

    def _fit(self, dataset: Dataset) -> 'DatasetTransformer':
        """
        Fit the transformer to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the transformer to.

        Returns
        -------
        self: DatasetTransformer
            The fitted transformer.
        """
        return self
