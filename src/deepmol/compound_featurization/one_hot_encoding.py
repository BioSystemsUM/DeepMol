import multiprocessing

import numpy as np

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.compound_featurization._utils import _PERIODIC_TABLE_ELEMENTS, _AVAILABLE_ELEMENTS


class SmilesOneHotEncoder(Transformer):
    """
    Encodes SMILES strings into a one-hot encoded matrix.

    Parameters
    ----------
    max_length: int
        The maximum length of the SMILES strings.
    n_jobs: int
        The number of jobs to run in parallel in the featurization.

    Attributes
    ----------
    dictionary: dict
        A dictionary mapping characters to integers.

    Examples
    --------
    >>> from deepmol.compound_featurization import SmilesOneHotEncoder
    >>> from deepmol.loaders import CSVLoader
    >>> data = loader = CSVLoader('data_path.csv', smiles_field='Smiles', labels_fields=['Class'])
    >>> dataset = loader.create_dataset(sep=";")
    >>> ohe = SmilesOneHotEncoder().fit_transform(dataset)
    """

    def __init__(self, max_length: int = 120, n_jobs: int = -1):
        """
        Initializes the featurizer.

        Parameters
        ----------
        max_length: int
            The maximum length of the SMILES strings.
        n_jobs: int
            The number of jobs to run in parallel in the featurization.
        """
        super().__init__()
        self.max_length = max_length + 1 # +1 for the padding
        self._available_chars = _AVAILABLE_ELEMENTS
        self._chars_to_replace = {}
        self.dictionary = {"": 0}
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()

    def featurize(self, dataset: Dataset) -> Dataset:
        """
        Featurizes a dataset.
        Computes the one-hot encoded matrix for each SMILES string in the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to featurize.

        Returns
        -------
        dataset: Dataset
            The featurized dataset.
        """
        return self._fit(dataset)._transform(dataset)

    def _fit(self, dataset: Dataset) -> 'SmilesOneHotEncoder':
        """
        Fits the featurizer.
        Computes the dictionary mapping characters to integers.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the featurizer on.

        Returns
        -------
        self: SmilesOneHotEncoder
            The fitted featurizer.
        """
        self._parse_two_char_tokens(list(dataset.smiles))
        self._set_up_dictionary(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        Computes the one-hot encoded matrix for each SMILES string in the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        smiles_matrix = self._encode_smiles_parallel(self._replace_chars(list(dataset.smiles)))
        dataset._X = np.array(smiles_matrix)
        return dataset

    def inverse_transform(self, dataset: Dataset) -> Dataset:
        """
        Inverse transforms the dataset.
        Computes the SMILES string for each one-hot encoded matrix in the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to inverse transform.

        Returns
        -------
        dataset: Dataset
            The inverse transformed dataset.
        """
        smiles = self._decode_smiles_parallel(list(dataset._X))
        dataset.smiles = np.array(smiles)
        return dataset

    def _encode(self, smiles: str) -> np.ndarray:
        """
        Encodes a SMILES string into a one-hot encoded matrix.

        Parameters
        ----------
        smiles: str
            The SMILES string to encode.

        Returns
        -------
        smiles_matrix: np.ndarray
            The one-hot encoded matrix.
        """
        smiles_matrix = np.zeros((len(self.dictionary), self.max_length))
        # TODO: remove smiles with length > max_length or do one hot encoding of smiles[:self.max_length] ?
        for index, char in enumerate(smiles[:self.max_length]):
            smiles_matrix[self.dictionary[char], index] = 1
        return smiles_matrix

    def _encode_smiles_parallel(self, smiles_list: list) -> list:
        """
        Encodes a list of SMILES strings into a list of one-hot encoded matrices.

        Parameters
        ----------
        smiles_list: list
            The list of SMILES strings to encode.

        Returns
        -------
        smiles_matrix_list: list
            The list of one-hot encoded matrices.
        """
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            encoded_smiles_list = pool.map(self._encode, smiles_list)
        return encoded_smiles_list

    def _decode(self, smiles_matrix: np.ndarray) -> str:
        """
        Decodes a one-hot encoded matrix into a SMILES string.

        Parameters
        ----------
        smiles_matrix: np.ndarray
            The one-hot encoded matrix to decode.

        Returns
        -------
        smiles: str
            The decoded SMILES string.
        """
        smiles = ''
        for row in smiles_matrix.T:
            char = list(self.dictionary.keys())[np.argmax(row)]
            smiles += char
            if char == '':
                break
        # replace back the two char tokens
        for comb, repl in self._chars_to_replace.items():
            smiles = smiles.replace(repl, comb)
        return smiles

    def _decode_smiles_parallel(self, smiles_matrix_list: list) -> list:
        """
        Decodes a list of one-hot encoded matrices into a list of SMILES strings using multiprocessing.

        Parameters
        ----------
        smiles_matrix_list: list
            The list of one-hot encoded matrices to decode.

        Returns
        -------
        smiles_list: list
            The list of decoded SMILES strings.
        """
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            decoded_smiles_list = pool.map(self._decode, smiles_matrix_list)
        return decoded_smiles_list

    def _parse_two_char_tokens(self, smiles: list) -> None:
        """
        Parses the two-char tokens in the SMILES strings.
        Replaces the two-char tokens with a single character (e.g. Br -> R). The replaced characters are added to the
        dictionary. We make sure that the replaced characters are not in the SMILES strings.

        Parameters
        ----------
        smiles: list
            The list of SMILES strings to parse.
        """
        combinations = set()
        for s in smiles:
            combinations.update(s[i:i + 2] for i in range(len(s) - 1) if s[i:i + 2].isalpha())
        self._chars_to_replace.update(
            {comb: self._available_chars.pop() for comb in combinations if comb in _PERIODIC_TABLE_ELEMENTS})

    def _set_up_dictionary(self, dataset: Dataset) -> None:
        """
        Sets up the dictionary mapping characters to integers. The characters are the ones in the SMILES strings and
        the replaced characters from the two-char tokens.

        Parameters
        ----------
        dataset: Dataset
            The dataset to set up the dictionary on.
        """
        processed_smiles = self._replace_chars(list(dataset.smiles))
        max_size = len(max(processed_smiles, key=len))
        if max_size < self.max_length:
            self.max_length = max_size + 1  # +1 for the padding
        self.dictionary.update({letter: idx for idx, letter in enumerate(set(''.join(processed_smiles)), start=1)})

    def _replace_chars(self, smiles: list) -> list:
        """
        Replaces the two-char tokens with a single character (e.g. Br -> R).

        Parameters
        ----------
        smiles: list
            The list of SMILES strings to replace the characters on.

        Returns
        -------
        processed_smiles: list
            The list of SMILES strings with the replaced characters.
        """
        processed_smiles = []
        for smile in smiles:
            sm = smile
            for comb, repl in self._chars_to_replace.items():
                sm = sm.replace(comb, repl)
            processed_smiles.append(sm)
        return processed_smiles

