from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, Mol

from deepmol.datasets import Dataset


class MolecularStandardizer(ABC):
    """
    Class for handling the standardization of molecules.
    """

    def __init__(self):
        """
        Standardizer for molecules.
        """
        if self.__class__ == MolecularStandardizer:
            raise Exception('Abstract class MolecularStandardizer should not be instantiated')

    def standardize(self, dataset: Dataset, log_every_n: int = 1000):
        """
        Standardizes a dataset of molecules.

        Parameters
        ----------
        dataset: Dataset
            Dataset to standardize.
        log_every_n: int
            Log every n molecules.

        Returns
        -------
        dataset: Dataset
            Standardized dataset.
        """
        molecules = dataset.mols

        stand_mols = []

        for i, mol in enumerate(molecules):
            molobj = None
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

                assert molobj is not None

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
    def _standardize(self, mol: Mol) -> Mol:
        """
        Standardizes a molecule.

        Parameters
        ----------
        mol: Mol
            RDKit Mol object

        Returns
        -------
        mol: Mol
            Standardized mol.
        """
        raise NotImplementedError
