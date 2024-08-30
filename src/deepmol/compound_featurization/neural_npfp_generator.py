import os

import numpy as np
from rdkit.Chem import Mol, MolToSmiles
import os

import pickle
import itertools

import pickle
from deepmol.compound_featurization.base_featurizer import MolecularFeaturizer

import torch

from .neural_npfp.model import MLP, FP_AE

from rdkit.Chem import AllChem
from rdkit import Chem

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class NeuralNPFP(MolecularFeaturizer):

    def __init__(self, model = "aux", device = "cpu", **kwargs) -> None:
        """
        Constructor for the Neural NPFP featurizer

        Parameters
        ----------
        model : str, optional
            The model to use, by default "aux"
        device : str, optional
            The device to run the model on, by default "cpu"
        """
        super().__init__(**kwargs)

        self.model = model

        self.device = device

        self.feature_names = [f'neural_npfp_{i}' for i in range(64)]


    def _featurize(self, mol: Mol):
        """
        Featurize a molecule using the neural npfp model

        Parameters
        ----------
        mol : Mol
            A molecule object
        
        Returns
        -------
        np.ndarray
            A 1D array of size 64

        """
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048)) 

        if self.model == "aux":
            model  = MLP([2048,1024,1024,64,49],1 ,0.2)
            model.load_state_dict(torch.load(os.path.join(FILE_PATH, "neural_npfp", "aux_cv0.pt"), map_location=torch.device("cpu")))
            model.eval()

        elif self.model == "base":
            model = MLP([2048,1024,1024,64,49],1 , 0.2)
            model.load_state_dict(torch.load(os.path.join(FILE_PATH,"neural_npfp", "baseline_cv0.pt"), map_location=torch.device("cpu")))
            model.eval()

        elif self.model == "ae":
            model = FP_AE([2048,512,64,512,2048], 1+True, 0.2)
            model.load_state_dict(torch.load(os.path.join(FILE_PATH,"neural_npfp","ae_cv0.pt"), map_location=torch.device("cpu")))
            model.eval()

        model.to(self.device)
        fp = torch.tensor([fp], dtype= torch.float)
        fp.to(self.device)
        _, _, nnfp = model(fp)
        nnfp = nnfp.detach().cpu().numpy()
        nnfp = nnfp.reshape(nnfp.shape[1])
        return nnfp