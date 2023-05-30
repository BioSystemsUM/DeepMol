import re
import warnings

from deepmol.datasets import Dataset
from deepmol.tokenizers import Tokenizer


class AtomLevelSmilesTokenizer(Tokenizer):
    """
    A tokenizer that tokenizes SMILES strings at the atom level (based on the SMILES grammar (regex)).

    Parameters
    ----------
    n_jobs: int
        The number of jobs to run in parallel in the tokenization.

    Attributes
    ----------
    _SMILES_REGEX: str
        The regex used to tokenize SMILES strings.
        The _SMILES_REGEX was taken from:
            [1] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas,
            and Alpha A. Lee ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated
            Chemical Reaction Prediction 1572-1583 DOI: 10.1021/acscentsci.9b00576

    Examples
    --------
    >>> from deepmol.tokenizers import AtomLevelSmilesTokenizer
    >>> from deepmol.loaders import CSVLoader

    >>> loader = CSVLoader('data_path.csv', smiles_field='Smiles', labels_fields=['Class'])
    >>> dataset = loader.create_dataset(sep=";")

    >>> tokenizer = AtomLevelSmilesTokenizer().fit(dataset)
    >>> tokens = tokenizer.tokenize(dataset)
    """
    _SMILES_REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self, n_jobs: int = -1):
        """
        Initializes the tokenizer.
        """
        super().__init__(n_jobs=n_jobs)
        self._regex = self._SMILES_REGEX
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
