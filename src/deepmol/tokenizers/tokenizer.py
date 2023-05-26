import multiprocessing
from abc import ABC, abstractmethod

import numpy as np


class Tokenizer(ABC):
    """
    An abstract class for tokenizers.
    Tokenizers are used to encode and decode strings.
    Child classes must implement the encode and decode methods and the shape property.
    """

    def __init__(self, n_jobs=-1):
        """
        Initializes the tokenizer.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel in the featurization.
        """
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()

    @abstractmethod
    def encode(self, smiles: str) -> np.ndarray:
        """
        Encodes a SMILES string into a one-hot encoded matrix.

        Parameters
        ----------
        smiles: str
            The SMILES string to encode.

        Returns
        -------
        smiles_matrix: np.ndarray
            The one-hot encoded matrix.
        """

    def _encode_smiles_parallel(self, smiles_list: list) -> list:
        """
        Encodes a list of SMILES strings into a list of one-hot encoded matrices.

        Parameters
        ----------
        smiles_list: list
            The list of SMILES strings to encode.

        Returns
        -------
        smiles_matrix_list: list
            The list of one-hot encoded matrices.
        """
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = [pool.apply_async(self.encode, args=(sm,)) for sm in smiles_list]
            encoded_smiles_list = [result.get() for result in results]
        return encoded_smiles_list

    @abstractmethod
    def decode(self, smiles_matrix: np.ndarray) -> str:
        """
        Decodes a one-hot encoded matrix into a SMILES string.

        Parameters
        ----------
        smiles_matrix: np.ndarray
            The one-hot encoded matrix to decode.

        Returns
        -------
        smiles: str
            The decoded SMILES string.
        """

    def _decode_smiles_parallel(self, smiles_matrix_list: list) -> list:
        """
        Decodes a list of one-hot encoded matrices into a list of SMILES strings using multiprocessing.

        Parameters
        ----------
        smiles_matrix_list: list
            The list of one-hot encoded matrices to decode.

        Returns
        -------
        smiles_list: list
            The list of decoded SMILES strings.
        """
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = [pool.apply_async(self.decode, args=(smiles_matrix,)) for smiles_matrix in smiles_matrix_list]
            decoded_smiles_list = [result.get() for result in results]
        return decoded_smiles_list

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        Returns the shape of the one-hot encoded matrix.
        """
