
import os
from typing import List, Union
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from deepmol.datasets.datasets import SmilesDataset

import torch
from rdkit.Chem import Mol
from deepmol.tokenizers import SmilesTokenizer

class LLMDataset(TorchDataset, SmilesDataset):

    def __init__(self,
                 smiles: Union[np.ndarray, List[str]],
                 mols: Union[np.ndarray, List[Mol]] = None,
                 ids: Union[List, np.ndarray] = None,
                 X: Union[List, np.ndarray] = None,
                 feature_names: Union[List, np.ndarray] = None,
                 y: Union[List, np.ndarray] = None,
                 label_names: Union[List, np.ndarray] = None,
                 mode: Union[str, List[str]] = 'auto',
                 max_length: int = 256, 
                 masking_probability: float = 0.15,
                 vocabulary_path: str = None,):
        """

        Parameters
        ----------
        tokenizer : BertTokenizer
            BertTokenizer
        smiles : Union[np.ndarray, List[str]]
            SMILES
        mols : Union[np.ndarray, List[Mol]], optional
            mols, by default None
        ids : Union[List, np.ndarray], optional
            Identifiers, by default None
        X : Union[List, np.ndarray], optional
            features, by default None
        feature_names : Union[List, np.ndarray], optional
            feature names, by default None
        y : Union[List, np.ndarray], optional
            label values, by default None
        label_names : Union[List, np.ndarray], optional
            label names, by default None
        mode : Union[str, List[str]], optional
            mode, by default 'auto'
        max_length : int, optional
            max length, by default 256
        masking_probability : float, optional
            Probability of masking item, by default 0.15
        """

        super().__init__(smiles, mols, ids, X, feature_names, y, label_names, mode)
        
        if vocabulary_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            SmilesTokenizer.export_vocab(self, os.path.join(dir_path, "vocab.txt"))
            self.tokenizer = SmilesTokenizer(os.path.join(dir_path, "vocab.txt"))
        else:
            self.tokenizer = SmilesTokenizer(vocabulary_path)
            
        self.masking_probability = masking_probability
        self.max_length = max_length

    def __len__(self):
        
        return len(self._smiles)

    def __getitem__(self, idx):
        smiles = self._smiles[idx]
        tokens = self.tokenizer(smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.squeeze()
        attention_mask = tokens.attention_mask.squeeze()
        
        # Create labels
        labels = input_ids.clone()
        
        # Masking
        probability_matrix = torch.full(labels.shape, self.masking_probability)  # 15% masking
        mask_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[mask_indices] = self.tokenizer.mask_token_id
        
        return input_ids, attention_mask, labels