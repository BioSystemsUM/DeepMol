from typing import List

import numpy as np

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.tokenizers import AtomLevelSmilesTokenizer, Tokenizer
from deepmol.utils.decorators import modify_object_inplace_decorator


class SmilesOneHotEncoder(Transformer):
    """
    A class for one-hot encoding SMILES.
    The SmilesOneHotEncoder tokenizes SMILES strings and one-hot encodes them.

    Parameters
    ----------
    tokenizer: Tokenizer
        The tokenizer to use to tokenize SMILES strings.
    max_length: int
        The maximum length of the SMILES strings.
    n_jobs: int
        The number of jobs to use for tokenization.

    Examples
    --------
    >>> from deepmol.compound_featurization import SmilesOneHotEncoder
    >>> from deepmol.loaders import CSVLoader

    >>> data = loader = CSVLoader('data_path.csv', smiles_field='Smiles', labels_fields=['Class'])
    >>> dataset = loader.create_dataset(sep=";")
    >>> ohe = SmilesOneHotEncoder().fit_transform(dataset)
    """

    def __init__(self, tokenizer: Tokenizer = None, max_length: int = None, n_jobs: int = -1):
        """
        Initializes the OneHotEncoder.

        Parameters
        ----------
        tokenizer: Tokenizer
            The tokenizer to use to tokenize SMILES strings.
        max_length: int
            The maximum length of the SMILES strings.
        n_jobs: int
            The number of jobs to use for tokenization.
        """
        super().__init__()
        self.tokenizer = AtomLevelSmilesTokenizer(n_jobs=n_jobs) if tokenizer is None else tokenizer
        self.max_length = max_length + 1 if max_length is not None else None
        self.n_jobs = n_jobs
        self.dictionary = {}

    def _fit(self, dataset: Dataset) -> 'SmilesOneHotEncoder':
        """
        Fits the featurizer.
        Computes the dictionary mapping characters to integers.

        Parameters
        ----------
        dataset: Dataset
            The dataset to featurize.

        Returns
        -------
        self: SmilesOneHotEncoder
            The fitted featurizer.
        """
        if not self.tokenizer.is_fitted():
            self.tokenizer.fit(dataset)
        self.dictionary = {token: i for i, token in enumerate(self.tokenizer.vocabulary)}
        # add "" and 'unk' to the dictionary
        self.dictionary.update({'': len(self.dictionary), 'unk': len(self.dictionary) + 1})
        if self.max_length is None:
            self.max_length = self.tokenizer.max_length + 1
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms a dataset.
        Computes the one-hot encoded matrix for each SMILES string in the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to featurize.

        Returns
        -------
        dataset: Dataset
            The one-hot-encoded dataset.
        """
        smiles = dataset.smiles
        multiprocessing_cls = JoblibMultiprocessing(process=self._one_hot_encode, n_jobs=self.n_jobs)
        one_hot = multiprocessing_cls.run(smiles)
        dataset.clear_cached_properties()
        dataset._X = np.array(one_hot)
        dataset.feature_names = [f'one_hot_{i}' for i in range(self.max_length)]
        return dataset

    @modify_object_inplace_decorator
    def featurize(self, dataset: Dataset) -> Dataset:
        """
        Featurizes a dataset (Fits and transforms).
        Computes the one-hot encoded matrix for each SMILES string in the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to featurize.

        Returns
        -------
        dataset: Dataset
            The one-hot-encoded dataset.
        """
        if self.is_fitted():
            return self.transform(dataset)
        return self.fit_transform(dataset)

    def _one_hot_encode(self, smiles: str) -> np.ndarray:
        """
        Computes the one-hot encoded matrix for a SMILES string.

        Parameters
        ----------
        smiles: str
            The SMILES string to one-hot encode.

        Returns
        -------
        one_hot: np.ndarray
            The one-hot encoded matrix.
        """
        smiles_matrix = np.zeros((len(self.dictionary), self.max_length))
        tokens = self.tokenizer._tokenize(smiles)
        for index, char in enumerate(tokens[:self.max_length]):
            if char not in self.dictionary.keys():
                char = 'unk'
            smiles_matrix[self.dictionary[char], index] = 1
        # fill the rest of the matrix with the padding token
        smiles_matrix[self.dictionary[''], len(tokens):] = 1
        return smiles_matrix

    def inverse_transform(self, matrix: np.ndarray) -> List[str]:
        """
        Inverse transforms a dataset.

        Parameters
        ----------
        matrix: np.ndarray
            The one-hot encoded matrix.

        Returns
        -------
        smiles: List[str]
            The SMILES strings.
        """
        multiprocessing_cls = JoblibMultiprocessing(process=self._decode, n_jobs=self.n_jobs)
        smiles = multiprocessing_cls.run(matrix)
        return list(smiles)

    def _decode(self, smiles_matrix: np.ndarray) -> str:
        """
        Decodes a one-hot encoded matrix.

        Parameters
        ----------
        smiles_matrix: np.ndarray
            The one-hot encoded matrix.

        Returns
        -------
        smiles: str
            The SMILES string.
        """
        smiles = []
        stride = self.tokenizer.stride if hasattr(self.tokenizer, 'stride') else 1
        tokenizer = self.tokenizer.atom_level_tokenizer \
            if hasattr(self.tokenizer, 'atom_level_tokenizer') else self.tokenizer
        tokens = []
        for row in smiles_matrix.T:
            char = list(self.dictionary.keys())[np.argmax(row)]
            if char == '':
                break
            tokens = tokenizer._tokenize(char)
            smiles.extend(tokens[:stride])
        smiles.extend(tokens[stride:])
        return "".join(smiles)

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the one-hot encoded matrix.

        Returns
        -------
        shape: tuple
            The shape of the one-hot encoded matrix.
        """
        return len(self.dictionary), self.max_length
