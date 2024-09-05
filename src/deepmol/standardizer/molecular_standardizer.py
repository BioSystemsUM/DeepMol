from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from rdkit.Chem import Mol

from deepmol.base import Transformer
from deepmol.datasets import Dataset
from deepmol.loggers.logger import Logger
from deepmol.parallelism.multiprocessing import JoblibMultiprocessing
from deepmol.utils.decorators import modify_object_inplace_decorator
from deepmol.utils.utils import canonicalize_mol_object, mol_to_smiles

from rdkit import RDLogger 


class MolecularStandardizer(ABC, Transformer):
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
        super().__init__()
        self.n_jobs = n_jobs
        RDLogger.DisableLog('rdApp.info')    
        self.logger = Logger()
        self.logger.info(f"Standardizer {self.__class__.__name__} initialized with {n_jobs} jobs.")

    def _standardize_mol(self, mol: Mol) -> Tuple[Mol, str]:
        """
        Standardizes a single molecule.

        Parameters
        ----------
        mol: Mol
            Molecule to standardize.

        Returns
        -------
        mol: Mol
            Standardized Mol object.
        smiles: str
            Standardized SMILES string.
        """
        try:
            mol_object = mol
            assert mol_object is not None
            mol_object = canonicalize_mol_object(mol_object)
            standardized_mol = self._standardize(mol_object)
            return standardized_mol, mol_to_smiles(standardized_mol, canonical=True)
        except Exception:
            return mol, mol_to_smiles(mol, canonical=True)

    @modify_object_inplace_decorator
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
        dataset._smiles = np.asarray([x[1] for x in result])
        dataset._mols = np.asarray([x[0] for x in result])
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

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Standardizes a dataset of molecules. This method is called by the `transform` method.

        Parameters
        ----------
        dataset: Dataset
            Dataset to standardize.

        Returns
        -------
        dataset: Dataset
            Standardized dataset.
        """
        return self.standardize(dataset)

    def _fit(self, dataset: Dataset) -> 'MolecularStandardizer':
        """
        Fits the standardizer to a dataset of molecules.

        Parameters
        ----------
        dataset: Dataset
            Dataset of molecules.

        Returns
        -------
        self: CustomStandardizer
            The fitted standardizer.
        """
        return self
