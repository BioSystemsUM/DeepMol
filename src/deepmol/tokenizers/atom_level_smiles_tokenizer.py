import re
import warnings

from deepmol.datasets import Dataset
from deepmol.tokenizers import Tokenizer
from deepmol.tokenizers._utils import _ATOM_LEVEL_SMILES_REGEX


class AtomLevelSmilesTokenizer(Tokenizer):
    """
    A tokenizer that tokenizes SMILES strings at the atom level (based on the SMILES grammar (regex)).

    Examples
    --------
    >>> from deepmol.tokenizers import AtomLevelSmilesTokenizer
    >>> from deepmol.loaders import CSVLoader

    >>> loader = CSVLoader('data_path.csv', smiles_field='Smiles', labels_fields=['Class'])
    >>> dataset = loader.create_dataset(sep=";")

    >>> tokenizer = AtomLevelSmilesTokenizer().fit(dataset)
    >>> tokens = tokenizer.tokenize(dataset)
    """

    def __init__(self, n_jobs: int = -1):
        """
        Initializes the tokenizer.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel. -1 means using all processors.
        """
        super().__init__(n_jobs=n_jobs)
        self._regex = _ATOM_LEVEL_SMILES_REGEX
        self._compiled_regex = None
        self._vocabulary = None
        self._max_length = None

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
    def max_length(self) -> int:
        """
        Returns the maximum length of the SMILES strings.

        Returns
        -------
        max_length: int
            The maximum length of the SMILES strings.
        """
        return self._max_length

    @property
    def regex(self) -> str:
        """
        Returns the regex used to tokenize SMILES strings.

        Returns
        -------
        regex: str
            The regex used to tokenize SMILES strings.
        """
        return self._regex

    @regex.setter
    def regex(self, regex: str):
        """
        Sets the regex used to tokenize SMILES strings.
        The tokenizer needs to be fitted again.

        Parameters
        ----------
        regex: str
            The regex used to tokenize SMILES strings.
        """
        self._regex = regex
        warnings.warn("The regex was changed. The tokenizer needs to be fitted again.")
        self._is_fitted = False

    def _fit(self, dataset: Dataset) -> 'AtomLevelSmilesTokenizer':
        """
        Fits the tokenizer to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the tokenizer to.

        Returns
        -------
        self: AtomLevelSmilesTokenizer
            The fitted tokenizer.
        """
        self._compiled_regex = re.compile(self.regex)
        self._is_fitted = True
        tokens = self.tokenize(dataset)
        self._vocabulary = list(set([token for sublist in tokens for token in sublist]))
        self._max_length = max([len(tokens) for tokens in tokens])
        return self

    def _tokenize(self, smiles: str) -> list:
        """
        Tokenizes a SMILES string.

        Parameters
        ----------
        smiles: str
            The SMILES string to tokenize.
        """
        return self._compiled_regex.findall(smiles)
