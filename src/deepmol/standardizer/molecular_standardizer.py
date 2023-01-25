from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, Mol

from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.utils.utils import canonicalize_mol_object


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

    def _standardize_mol(self, mol: Mol) -> Mol:
        mol_object = None
        try:
            if isinstance(mol, str):
                # mol must be a RDKit Mol object, so parse a SMILES
                mol_object = Chem.MolFromSmiles(mol)
                mol_object = canonicalize_mol_object(mol_object)
            elif isinstance(mol, Mol):
                mol_object = canonicalize_mol_object(mol)

            assert mol_object is not None

            return Chem.MolToSmiles(self._standardize(mol_object), canonical=True)
        except Exception as e:
            if isinstance(mol, Chem.rdchem.Mol):
                mol = Chem.MolToSmiles(mol, canonical=True)
            return mol

    def standardize(self, dataset: Dataset, n_jobs=-1):
        """
        Standardizes a dataset of molecules.

        Parameters
        ----------
        dataset: Dataset
            Dataset to standardize.
        n_jobs: int
            Number of jobs to use for parallelization. If -1, all available cores are used.

        Returns
        -------
        dataset: Dataset
            Standardized dataset.
        """
        molecules = dataset.mols

        multiprocessing_cls = JoblibMultiprocessing(n_jobs=n_jobs, process=self._standardize_mol)

        result = list(multiprocessing_cls.run(molecules))

        dataset.mols = np.asarray(result)

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
