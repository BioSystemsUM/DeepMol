from abc import ABC, abstractmethod

from rdkit import Chem
from Datasets.Datasets import Dataset
from rdkit.Chem import rdmolfiles, Mol
from rdkit.Chem import rdmolops
import numpy as np


class MolecularStandardizer(ABC):
    """
    Class for handling the standardization of molecules.
    """

    def __init__(self):
        """Standardizer for molecules.
        Parameters
        ----------
        Returns
        -------
        dataset: Dataset object
          The input Dataset containing a standardized representation of the molecules in Dataset.mols.
        """

        if self.__class__ == MolecularStandardizer:
            raise Exception('Abstract class MolecularStandardizer should not be instantiated')

    def standardize(self, dataset: Dataset, log_every_n=1000):
        molecules = dataset.mols

        stand_mols = []

        for i, mol in enumerate(molecules):
            if i % log_every_n == 0:
                print("Standardizing datapoint %i" % i)
            try:
                if isinstance(mol, str):
                    # mol must be a RDKit Mol object, so parse a SMILES
                    molobj = Chem.MolFromSmiles(mol)
                    try:
                        # SMILES is unique, so set a canonical order of atoms
                        new_order = rdmolfiles.CanonicalRankAtoms(molobj)
                        molobj = rdmolops.RenumberAtoms(molobj, new_order)
                    except Exception as e:
                        molobj = mol
                elif isinstance(mol, Mol):
                    try:
                        molobj = mol
                        # SMILES is unique, so set a canonical order of atoms
                        new_order = rdmolfiles.CanonicalRankAtoms(molobj)
                        molobj = rdmolops.RenumberAtoms(molobj, new_order)
                    except Exception as e:
                        molobj = mol

                stand_mols.append(Chem.MolToSmiles(self._standardize(molobj)))
            except Exception as e:
                if isinstance(mol, Chem.rdchem.Mol):
                    mol = Chem.MolToSmiles(mol)
                print("Failed to featurize datapoint %d, %s. Appending non standardized mol" % (i, mol))
                print("Exception message: {}".format(e))
                stand_mols.append(mol)
        dataset.mols = np.asarray(stand_mols)

        return dataset

    @abstractmethod
    def _standardize(self, mol: Mol):
        raise NotImplementedError
