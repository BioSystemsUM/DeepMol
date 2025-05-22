import random
from typing import List, Union
import numpy as np
import torch
import torch
from transformers import BertTokenizer

import re
from tqdm import tqdm
from deepmol.datasets.datasets import Dataset
from deepmol.tokenizers.tokenizer import Tokenizer
from transformers import BertTokenizer

from rdkit import Chem
from rdkit.Chem import AllChem
from deepmol.utils.utils import canonicalize_mol_object




class NPTokenizer(Tokenizer):
    """Run regex tokenization"""

    def __init__(self, n_jobs=-1) -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
            suffix: optional suffix for the tokens. Defaults to "".
        """
        super().__init__(n_jobs)
        self._vocabulary = None
        self._max_length = None

    def generate_sentence(self, smiles):# max_length = 2032
        
        try:
            mol = Chem.MolFromSmiles(smiles) # conver smiles to mol
            mol = canonicalize_mol_object(mol)
        except:
            return None
        if mol is None:
            return None
        atom_info = {}
        info = {}
        _ = AllChem.GetMorganFingerprint(mol, 1, bitInfo=info)
        for key, tup in info.items():
            for t in tup:
                atom_index = t[0]
                if atom_index not in atom_info:
                    atom_info[atom_index] = []
                # r = t[1]
                k = str(key)
                atom_info[atom_index].append(k)

        sentence = []
        for atom in sorted(atom_info):
            sentence.extend(atom_info[atom]) 

        return sentence

    def _tokenize(self, text: str):
        """Regex tokenization.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens separated by spaces.
        """
        tokens = self.generate_sentence(text)
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
        
    
    def _fit(self, dataset: Dataset) -> 'NPBERTTokenizer':
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

class NPBERTTokenizer(BertTokenizer):
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
        self.tokenizer = NPTokenizer.from_file(vocab_file)
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
    def export_vocab(dataset, output_path, export_tokenizer=False):
        unique_tokens = set()

        tokenizer = NPTokenizer()
        tokenizer.fit(dataset=dataset)

        unique_tokens = tokenizer.vocabulary
        unk_token: str = "[UNK]"
        sep_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        cls_token: str = "[CLS]"
        mask_token: str = "[MASK]"

        unique_tokens = [pad_token, unk_token, sep_token, mask_token, cls_token] + unique_tokens
        print("Exporting vocabulary to", output_path)
        with open(output_path, "w") as f:
            for token in unique_tokens:
                f.write(token + "\n")

        if export_tokenizer:
            tokenizer.to_pickle("tokenizer.pkl")

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