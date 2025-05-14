
import os
import random
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
                 vocabulary_path: str = None,
                 mask: bool = True):
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
        mask : bool, optional
            Whether the dataset will be used for masked learning. If False, it assumes it will be used for fine-tuning
        """

        super().__init__(smiles, mols, ids, X, feature_names, y, label_names, mode)
        
        if vocabulary_path is None:
            dir_path = os.getcwd()
            SmilesTokenizer.export_vocab(self, os.path.join(dir_path, "vocab.txt"))
            self.tokenizer = SmilesTokenizer(os.path.join(dir_path, "vocab.txt"))
        else:
            self.tokenizer = SmilesTokenizer(vocabulary_path)
        
        self.masking_probability = masking_probability
        self.max_length = max_length
        self.mask = mask

    def __len__(self):
        
        return len(self._smiles)

    def __getitem__(self, idx):
        smiles = self._smiles[idx]
        tokens = self.tokenizer(smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.squeeze()
        attention_mask = tokens.attention_mask.squeeze()
        mask_indices = None

        if self.mask:
            torch.manual_seed(42)
            random.seed(42)
            labels = input_ids.clone()
            
            # Identify special tokens
            special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.unk_token_id, 
                            self.tokenizer.sep_token_id, self.tokenizer.mask_token_id, 
                            self.tokenizer.cls_token_id}
            
            # Create masking probability matrix
            probability_matrix = torch.full(labels.shape, self.masking_probability)
            mask_indices = torch.bernoulli(probability_matrix).bool()
            
            # Avoid masking special tokens
            for token_id in special_tokens:
                mask_indices &= input_ids != token_id
            
            # Ensure masked tokens are valid (select other tokens if special)
            for idx in torch.where(mask_indices)[0]:
                if input_ids[idx].item() in special_tokens:
                    available_tokens = [t for t in range(self.tokenizer.vocab_size) if t not in special_tokens]
                    input_ids[idx] = torch.tensor(random.choice(available_tokens), dtype=torch.long)
            
            input_ids[mask_indices] = self.tokenizer.mask_token_id
            return input_ids, attention_mask, labels, mask_indices
        
        else:
            if self.y is not None:
                if len(self.y.shape) == 1:
                    labels = torch.tensor(self.y[idx], dtype=torch.long)
                else:
                    labels = torch.tensor(self.y[idx, :], dtype=torch.float)
            else:
                labels = torch.tensor([])

            return input_ids, attention_mask, labels
