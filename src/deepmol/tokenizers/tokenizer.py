from abc import ABC, abstractmethod

from deepmol.base import Estimator
from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing


class Tokenizer(Estimator, ABC):
    """
    An abstract class for tokenizers.
    Tokenizers are used to tokenize strings.
    Child classes must implement the tokenize method.
    """

    def __init__(self, n_jobs: int) -> None:
        """
        Initializes the tokenizer.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel in the featurization.
        """
        super().__init__()
        self.n_jobs = n_jobs

    def tokenize(self, dataset: Dataset) -> list:
        """
        Tokenizes a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to tokenize.

        Returns
        -------
        dataset: Dataset
            The tokenized dataset.
        """
        if not self._is_fitted:
            raise ValueError("The tokenizer must be fitted before tokenizing a dataset. "
                             "Call Tokenizer.fit(dataset) first.")
        smiles = dataset.smiles
        multiprocessing_cls = JoblibMultiprocessing(process=self._tokenize, n_jobs=self.n_jobs)
        tokens = multiprocessing_cls.run(smiles)
        return list(tokens)

    @abstractmethod
    def _tokenize(self, text: str) -> list:
        """
        Tokenizes a text.

        Parameters
        ----------
        text: str
            The text to tokenize.

        Returns
        -------
        tokens: list
            The list of tokens.
        """

    @property
    @abstractmethod
    def vocabulary(self) -> list:
        """
        Returns the vocabulary.

        Returns
        -------
        vocabulary: list
            The vocabulary.
        """

    @property
    @abstractmethod
    def max_length(self) -> int:
        """
        Returns the maximum length of a tokenized string.

        Returns
        -------
        max_length: int
            The maximum length of a tokenized string.
        """
