import re
import warnings

from deepmol.datasets import Dataset
from deepmol.tokenizers import Tokenizer, AtomLevelSmilesTokenizer
from deepmol.tokenizers._utils import _ATOM_LEVEL_SMILES_REGEX


class KmerSmilesTokenizer(Tokenizer):

    def __init__(self, size: int = 3, stride: int = 1, n_jobs: int = -1):
        """
        Initializes the tokenizer.

        Parameters
        ----------
        size: int
            The size of the k-mers.
        stride: int
            The stride of the k-mers (distance between the starting positions of consecutive tokens in the sequence).
        n_jobs: int
            The number of jobs to run in parallel. -1 means using all processors.
        """
        super().__init__(n_jobs=n_jobs)
        self._size = size
        self._stride = stride
        self._regex = _ATOM_LEVEL_SMILES_REGEX
        self._compiled_regex = None
        self._fitted_atom_level_tokenizer = None
        self._vocabulary = None
        self._max_length = None

    def _fit(self, dataset: Dataset) -> 'KmerSmilesTokenizer':
        """
        Fits the tokenizer to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the tokenizer to.

        Returns
        -------
        self: KmerSmilesTokenizer
            The fitted tokenizer.
        """
        self._compiled_regex = re.compile(self.regex)
        self._fitted_atom_level_tokenizer = AtomLevelSmilesTokenizer().fit(dataset)
        units = self._fitted_atom_level_tokenizer.vocabulary
        if self._size == 1:
            tokens = units
            max_len = self._fitted_atom_level_tokenizer.max_length
        else:
            tokens = set()
            max_len = 0
            for smile in dataset.smiles:
                tkns = self._fitted_atom_level_tokenizer._tokenize(smile)
                tkns = ["".join(tkns[i:i + self._size]) for i in range(0, len(tkns), self._stride)
                        if i + self._size <= len(tkns)]
                tokens.update(tkns)
                if len(tkns) > max_len:
                    max_len = len(tkns)
        self._vocabulary = tokens
        self._max_length = max_len
        return self

    def _tokenize(self, smiles: str) -> list:
        """
        Tokenizes a SMILES string.

        Parameters
        ----------
        smiles: str
            The SMILES string to tokenize.

        Returns
        -------
        tokens: list
            The tokens of the SMILES string.
        """
        tkns = self._fitted_atom_level_tokenizer._tokenize(smiles)
        tkns = ["".join(tkns[i:i + self._size]) for i in range(0, len(tkns), self._stride)
                if i + self._size <= len(tkns)]
        return tkns

    @property
    def max_length(self) -> int:
        """
        Returns the maximum length (maximum number of tokens) of the SMILES strings.

        Returns
        -------
        max_length: int
            The maximum length of the SMILES strings.
        """
        return self._max_length

    @property
    def vocabulary(self) -> list:
        """
        Returns the vocabulary of the tokenizer.

        Returns
        -------
        vocabulary: list
            The vocabulary of the tokenizer.
        """
        return self._vocabulary

    @property
    def regex(self) -> str:
        """
        Returns the regex used to tokenize the SMILES strings.

        Returns
        -------
        regex: str
            The regex used to tokenize the SMILES strings.
        """
        return self._regex

    @regex.setter
    def regex(self, regex: str) -> None:
        """
        Sets the regex used to tokenize the SMILES strings.

        Parameters
        ----------
        regex: str
            The regex used to tokenize the SMILES strings.
        """
        self._regex = regex
        warnings.warn("The regex was changed. The tokenizer needs to be fitted again.")
        self._is_fitted = False

    @property
    def size(self) -> int:
        """
        Returns the size of the k-mers.

        Returns
        -------
        size: int
            The size of the k-mers.
        """
        return self._size

    @property
    def stride(self) -> int:
        """
        Returns the stride of the k-mers (overlap between consecutive k-mers).

        Returns
        -------
        stride: int
            The stride of the k-mers.
        """
        return self._stride

    @property
    def atom_level_tokenizer(self) -> AtomLevelSmilesTokenizer:
        """
        Returns the fitted atom-level tokenizer used to tokenize the SMILES strings.

        Returns
        -------
        atom_level_tokenizer: AtomLevelSmilesTokenizer
            The fitted atom-level tokenizer used to tokenize the SMILES strings.
        """
        return self._fitted_atom_level_tokenizer
