from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from deepmol.datasets import Dataset
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.utils.utils import canonicalize_mol_object


class MolecularStandardizer(ABC):
    """
    Class for handling the standardization of molecules.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        """
        Standardizer for molecules.

        Parameters
        ----------
        n_jobs: int
            Number of jobs to run in parallel.
        """
        self.n_jobs = n_jobs

    def _standardize_mol(self, mol: Union[Mol, str]) -> Mol:
        """
        Standardizes a single molecule.

        Parameters
        ----------
        mol: Union[Mol, str]
            Molecule to standardize.

        Returns
        -------
        mol: str
            Standardized SMILES string.
        """
        try:
            mol_object = mol
            if isinstance(mol, str):
                # mol must be a RDKit Mol object, so parse a SMILES
                mol_object = Chem.MolFromSmiles(mol_object)

            assert mol_object is not None
            mol_object = canonicalize_mol_object(mol_object)

            return Chem.MolToSmiles(self._standardize(mol_object), canonical=True)
        except Exception:
            if isinstance(mol, Chem.rdchem.Mol):
                mol = Chem.MolToSmiles(mol, canonical=True)
            return mol

    def standardize(self, dataset: Dataset) -> Dataset:
        """
        Standardizes a dataset of molecules.

        Parameters
        ----------
        dataset: Dataset
            Dataset to standardize.

        Returns
        -------
        dataset: Dataset
            Standardized dataset.
        """
        molecules = dataset.mols
        multiprocessing_cls = JoblibMultiprocessing(n_jobs=self.n_jobs, process=self._standardize_mol)
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
