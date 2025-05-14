import os
from typing import Tuple
import numpy as np
from rdkit.Chem import Mol
import torch
from tqdm import tqdm

from deepmol.compound_featurization import MolecularFeaturizer
from transformers import BertTokenizer, BertModel, BertConfig

from deepmol.tokenizers.transformers_tokenizer import SmilesTokenizer
from deepmol.utils.decorators import modify_object_inplace_decorator
from deepmol.utils.errors import PreConditionViolationException

from deepmol.datasets import Dataset

class LLM(MolecularFeaturizer):
    """
    MHFP featurizer class. This module contains the MHFP encoder, which is used to encode SMILES and RDKit
    molecule instances as MHFP fingerprints.
    """

    def __init__(self, model_path: str, tokenizer: BertTokenizer = None, model = BertModel, config_class = BertConfig,**kwargs):
        super().__init__(**kwargs)

        # read bert model
        if tokenizer is None:
            tokenizer = SmilesTokenizer(vocab_file=os.path.join(model_path, "vocab.txt"))
        self.tokenizer = tokenizer
        self.config = config_class.from_json_file(os.path.join(model_path, "config.json"))
        self.model = model.from_pretrained(os.path.join(model_path, "model.pt"), config=self.config)

        self.feature_names = [f'llm_{i}' for i in range(self.config.output_hidden_states)]

    @modify_object_inplace_decorator
    def featurize(self,
                  dataset: Dataset,
                  remove_nans_axis: int = 0
                  ) -> Dataset:

        """
        Calculate features for molecules.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the molecules to featurize in dataset.mols.
        remove_nans_axis: int
            The axis to remove NaNs from. If None, no NaNs are removed.

        Returns
        -------
        dataset: Dataset
          The input Dataset containing a featurized representation of the molecules in Dataset.X.
        """
        smiles = dataset.smiles
        remove_mols = []
        features = []
        for smile in tqdm(smiles):
            feat, remove_mol = self._featurize_mol(smile)
            features.append(feat)
            remove_mols.append(remove_mol)

        remove_mols_list = np.array(remove_mols)
        dataset.remove_elements(np.array(dataset.ids)[remove_mols_list], inplace=True)
        
        features = np.array(features, dtype=object)
        features = features[~remove_mols_list]
        
        try:
            features = features.astype('float64')
        except:
            pass

        if (isinstance(features[0], np.ndarray) and len(features[0].shape) == 2) or not isinstance(features[0],
                                                                                                   np.ndarray):
            pass
        else:
            features = np.vstack(features)

        dataset.clear_cached_properties()
        dataset._X = features
        dataset.feature_names = self.feature_names
        dataset.remove_nan(remove_nans_axis, inplace=True)
        return dataset
    
    def _featurize_mol(self, mol: str) -> Tuple[np.ndarray, bool]:
        """
        Calculate features for a single molecule.

        Parameters
        ----------
        mol: string
            The molecule to featurize.

        Returns
        -------
        features: np.ndarray
            The features for the molecule.
        remove_mol: bool
            Whether the molecule should be removed from the dataset.
        """
        try:
            feat = self._featurize(mol)
            remove_mol = False
            return feat, remove_mol
        except PreConditionViolationException:
            exit(1)

        except Exception as e:
            smiles = None
            self.logger.error(f"Failed to featurize {smiles}. Appending empty array")
            self.logger.error("Exception message: {}".format(e))
            remove_mol = True
            return np.array([]), remove_mol

    def _featurize(self, mol: str):
        # Tokenize the text
        inputs = self.tokenizer(mol, max_length=self.config.max_position_embeddings, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Get the model's output
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Get the mean of the last hidden state
        mean_last_hidden_state = last_hidden_state[:, 1:, :].mean(dim=1)

        # Convert to numpy array
        mean_last_hidden_state = mean_last_hidden_state.squeeze()
        features = mean_last_hidden_state.detach().numpy()
        return features
