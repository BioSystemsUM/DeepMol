import numpy as np

#from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.compound_featurization._utils import _PERIODIC_TABLE_ELEMENTS, _AVAILABLE_ELEMENTS


class SmilesOneHotEncoder:#(Transformer):

    def __init__(self, max_length: int = 120, **kwargs):
        #super().__init__()
        self.max_length = max_length + 1 # +1 for the padding
        self._available_chars = _AVAILABLE_ELEMENTS
        self._chars_to_replace = {}
        self.dictionary = {"": 0}

    def _fit(self, dataset: Dataset) -> 'SmilesOneHotEncoder':
        self._parse_two_char_tokens(list(dataset.smiles))
        self._set_up_dictionary(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        smiles_matrix = np.array([self._encode(smile) for smile in self._replace_chars(list(dataset.smiles))])
        dataset._X = smiles_matrix
        return dataset

    def _inverse_transform(self, dataset: Dataset) -> Dataset:
        smiles = []
        for row in dataset._X:
            smiles.append(self._decode(row))
        dataset.smiles = np.array(smiles)
        return dataset

    def _encode(self, smiles: str) -> np.ndarray:
        smiles_matrix = np.zeros((len(self.dictionary), self.max_length))
        # TODO: remove smiles with length > max_length or do one hot encoding of smiles[:self.max_length] ?
        for index, char in enumerate(smiles[:self.max_length]):
            smiles_matrix[self.dictionary[char], index] = 1
        return smiles_matrix

    def _decode(self, smiles_matrix: np.ndarray) -> str:
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

    def _parse_two_char_tokens(self, smiles: list) -> None:
        combinations = set()
        for s in smiles:
            combinations.update(s[i:i + 2] for i in range(len(s) - 1) if s[i:i + 2].isalpha())
        self._chars_to_replace.update(
            {comb: self._available_chars.pop() for comb in combinations if comb in _PERIODIC_TABLE_ELEMENTS})

    def _set_up_dictionary(self, dataset: Dataset) -> None:
        processed_smiles = self._replace_chars(list(dataset.smiles))
        max_size = len(max(processed_smiles, key=len))
        if max_size < self.max_length:
            self.max_length = max_size + 1  # +1 for the padding
        self.dictionary.update({letter: idx for idx, letter in enumerate(set(''.join(processed_smiles)), start=1)})

    def _replace_chars(self, smiles: list) -> list:
        processed_smiles = []
        for smile in smiles:
            sm = smile
            for comb, repl in self._chars_to_replace.items():
                sm = sm.replace(comb, repl)
            processed_smiles.append(sm)
        return processed_smiles

