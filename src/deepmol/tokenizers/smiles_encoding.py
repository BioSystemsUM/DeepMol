import re
from typing import List

import numpy as np

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.tokenizers import Tokenizer
from deepmol.tokenizers._utils import _SMILES_TOKENS, _AVAILABLE_TOKENS


class SmilesOneHotEncoder(Transformer, Tokenizer):
    """
    Encodes SMILES strings into a one-hot encoded matrix.
    The SmilesCharacterLevelTokenizer treats every character is the SMILES as a single token.
    However, two-charater elements such Al and Br, Cl, etc. are treated as a single token.

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
    >>> from deepmol.tokenizers import SmilesOneHotEncoder
    >>> from deepmol.loaders import CSVLoader
    >>> data = loader = CSVLoader('data_path.csv', smiles_field='Smiles', labels_fields=['Class'])
    >>> dataset = loader.create_dataset(sep=";")
    >>> ohe = SmilesOneHotEncoder().fit_transform(dataset)
    """

    def __init__(self, max_length: int = None, regex: bool = True, n_jobs: int = -1):
        """
        Initializes the featurizer.

        Parameters
        ----------
        max_length: int
            The maximum length of the SMILES strings.
        regex: bool
            Whether to use regex to replace characters. Treats '[*]' as one token.
        n_jobs: int
            The number of jobs to run in parallel in the featurization.
        """
        Transformer.__init__(self)
        Tokenizer.__init__(self, n_jobs=n_jobs)
        self.max_length = max_length + 1 if max_length is not None else None
        self.regex = regex
        self._available_chars = _AVAILABLE_TOKENS
        self.processed_smiles = []
        self._chars_to_replace = {}
        self.dictionary = {}

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
        if self.is_fitted():
            return self.transform(dataset)
        return self.fit_transform(dataset)

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
        if self.regex:
            self._parse_regex_tokens(list(dataset.smiles))
            self.processed_smiles = self._replace_chars(list(dataset.smiles))
        self._parse_two_char_tokens(list(dataset.smiles))
        self.processed_smiles = self._replace_chars(list(dataset.smiles))
        self._set_up_dictionary()
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

    def encode(self, smiles: str) -> np.ndarray:
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
            if char not in self.dictionary.keys():
                char = 'unk'
            smiles_matrix[self.dictionary[char], index] = 1
        # fill the rest of the matrix with the padding token
        smiles_matrix[self.dictionary[''], len(smiles):] = 1
        return smiles_matrix

    def decode(self, smiles_matrix: np.ndarray) -> str:
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

    def _parse_two_char_tokens(self, smiles: List[str]) -> None:
        """
        Parses the two-char tokens in the SMILES strings.
        Replaces the two-char tokens with a single character (e.g. Br -> R). The replaced characters are added to the
        dictionary. We make sure that the replaced characters are not in the SMILES strings.

        Parameters
        ----------
        smiles: List[str]
            The list of SMILES strings to parse.
        """
        combinations = set()
        for s in smiles:
            combinations.update(s[i:i + 2] for i in range(len(s) - 1) if s[i:i + 2].isalpha())
        self._chars_to_replace.update(
            {comb: self._available_chars.pop() for comb in combinations if comb in _SMILES_TOKENS})

    def _parse_regex_tokens(self, smiles: List[str]) -> None:
        """
        Parses the regex tokens in the SMILES strings.
        Replaces the regex tokens with a single character (e.g. [C@@H] -> R). The replaced characters are added to the
        dictionary. We make sure that the replaced characters are not in the SMILES strings.

        Parameters
        ----------
        smiles: List[str]
            The list of SMILES strings to parse.
        """
        regex = "(\[[^\[\]]{1,6}\])"
        finds = []
        for s in smiles:
            char_list = re.split(regex, s)
            for char in char_list:
                if char.startswith('['):
                    finds.append(char)
        self._chars_to_replace.update({comb: self._available_chars.pop() for comb in set(finds)})

    def _set_up_dictionary(self) -> None:
        """
        Sets up the dictionary mapping characters to integers. The characters are the ones in the SMILES strings and
        the replaced characters from the two-char tokens.
        """
        max_size = len(max(self.processed_smiles, key=len))
        if self.max_length is None or max_size < self.max_length:
            self.max_length = max_size + 1  # +1 for the padding
        self.dictionary.update({letter: idx for idx, letter in enumerate(set(''.join(self.processed_smiles)))})
        # add "" and 'unk' to the dictionary
        self.dictionary.update({'': len(self.dictionary), 'unk': len(self.dictionary) + 1})

    def _replace_chars(self, smiles: List[str]) -> List[str]:
        """
        Replaces the two-char tokens with a single character (e.g. Br -> R).

        Parameters
        ----------
        smiles: : List[str]
            The list of SMILES strings to replace the characters on.

        Returns
        -------
        processed_smiles: : List[str]
            The list of SMILES strings with the replaced characters.
        """
        processed_smiles = []
        for smile in smiles:
            sm = smile
            for comb, repl in self._chars_to_replace.items():
                sm = sm.replace(comb, repl)
            processed_smiles.append(sm)
        return processed_smiles

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
