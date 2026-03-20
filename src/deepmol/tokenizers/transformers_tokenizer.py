import re
from tqdm import tqdm
from deepmol.datasets.datasets import Dataset
from deepmol.tokenizers.tokenizer import Tokenizer
from transformers import BertTokenizer

class RegexTokenizer(Tokenizer):
    """Run regex tokenization"""

    def __init__(self, n_jobs=-1) -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
            suffix: optional suffix for the tokens. Defaults to "".
        """
        super().__init__(n_jobs)
        self.regex_pattern = r"(\%\([0-9]{3}\)|\[[^\]]+]|Se?|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.regex_pattern)
        self._vocabulary = None
        self._max_length = None

    def _tokenize(self, text: str):
        """Regex tokenization.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens separated by spaces.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens
    
    @classmethod
    def from_file(cls, file_path: str):

        with open(file_path, mode="r") as f:
            lines = f.readlines()
            vocabulary = list(set([token.strip() for token in lines]))

        new_tokenizer = cls()
        new_tokenizer._vocabulary = vocabulary
        new_tokenizer._is_fitted = True
        return new_tokenizer
        
    
    def _fit(self, dataset: Dataset) -> 'RegexTokenizer':
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
    def vocabulary(self) -> list:
        """
        Returns the vocabulary of the tokenizer.

        Returns
        -------
        vocabulary: list
            The vocabulary of the tokenizer.
        """
        return self._vocabulary

class SmilesTokenizer(BertTokenizer):
    """
    Constructs a SmilesTokenizer.
    Adapted from https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp.

    Args:
        vocabulary_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file: str,
        do_lower_case=False,
        **kwargs,
    ) -> None:
        """Constructs an SmilesTokenizer.
        Args:
            vocabulary_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
        """
        unk_token: str = "[UNK]"
        sep_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        cls_token: str = "[CLS]"
        mask_token: str = "[MASK]"
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        # define tokenization utilities
        self.tokenizer = RegexTokenizer.from_file(vocab_file)
        self.unique_tokens = [pad_token, unk_token, sep_token, cls_token, mask_token]

    @property
    def vocab_list(self):
        """List vocabulary tokens.
        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str):
        """Tokenize a text representing a SMILES
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """
        return self.tokenizer._tokenize(text)

    @staticmethod
    def export_vocab(dataset, output_path):
        unique_tokens = set()

        tokenizer = RegexTokenizer()
        tokenizer.fit(dataset=dataset)

        unique_tokens = tokenizer.vocabulary
        unk_token: str = "[UNK]"
        sep_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        cls_token: str = "[CLS]"
        mask_token: str = "[MASK]"

        unique_tokens = [pad_token, unk_token, sep_token, mask_token, cls_token] + unique_tokens
        with open(output_path, "w") as f:
            for token in unique_tokens:
                f.write(token + "\n")

    def get_max_size(self, smiles_list):
        
        max_size = 0
        for smile in tqdm(smiles_list, total=len(smiles_list)):
            tokens = self.tokenizer._tokenize(smile)
            if len(tokens) > max_size:
                max_size = len(tokens)
        
        return max_size
    
    def get_all_sizes(self, smiles_list):

        lengths = []
        for smile in tqdm(smiles_list, total=len(smiles_list)):
            tokens = self.tokenizer._tokenize(smile)
            lengths.append(len(tokens))

        return lengths
