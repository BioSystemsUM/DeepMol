from deepmol.compound_featurization.base_featurizer import MolecularFeaturizer
from deepmol.datasets.datasets import SmilesDataset
from .nc_mfp.Scaffold_matching_step import ScaffoldMatching
from .nc_mfp.SFCP_assigning_step import SFCPAssigning
from .nc_mfp.Fragment_identifying_step import FragmentIdentifying
from .nc_mfp.Fingerprint_representation_step import FingerprintRepresentation
from .nc_mfp import NC_MFP_PATH

import numpy as np
from rdkit.Chem import Mol, MolToSmiles
import os

import pickle
import itertools


class NcMfp(MolecularFeaturizer):

    def __init__(self, database_file_path=os.path.join(NC_MFP_PATH, 'databases', 'ncdb'), **kwargs) -> None:
        """
        NC MFP fingerprint generator.

        All credits to the authors of the original code:
        Myungwon Seo, Hyun Kil Shin, Yoochan Myung, Sungbo Hwang & Kyoung Tai No

        Publication: https://doi.org/10.1186/s13321-020-0410-3

        Parameters
        ----------
        database_file_path: str
            The path to the database file
        """
        super().__init__(**kwargs)

        self.all_scaffold_file_path = (os.path.join(NC_MFP_PATH, "databases", 'All_Optimized_Scaffold_List.txt'))
        # import Final_all_Fragment_Dic from pickle
        with open(os.path.join(database_file_path, 'fragment_dic.pickle'), 'rb') as f:
            self.final_all_fragments_Dic = pickle.load(f)

        with open(os.path.join(database_file_path, 'final_label.pickle'), 'rb') as f:
            self.final_all_nc_mfp_labels = pickle.load(f)

        self.feature_names = [f'nc_mfp_{i}' for i in range(len(self.final_all_nc_mfp_labels))]

    def _featurize(self, mol: Mol) -> np.ndarray:
        """
        Calculate morgan fingerprint for a single molecule.

        Parameters
        ----------
        mol: Mol
          RDKit Mol object

        Returns
        -------
        fp: np.ndarray
          A numpy array of circular fingerprint.
        """
        step_2 = ScaffoldMatching()
        step_4 = SFCPAssigning()
        step_5 = FragmentIdentifying()
        step_6 = FingerprintRepresentation()

        mol = MolToSmiles(mol)

        # [2] Scaffold matching step #
        final_all_scaffold_match_list = [step_2.match_all_scaffold_smarts(self.all_scaffold_file_path, mol)]

        # Merge scaffold lists
        final_all_scaffold_match_list = list(itertools.chain(*final_all_scaffold_match_list))
        final_all_scaffold_match_list = list(set(final_all_scaffold_match_list))

        # [3] Fragment list generation & [4] Scfffold-Fragment Connection List(SFCP) assigning step #
        final_all_sfcp_list = [step_4.assign_All_SFCP_Smarts(self.all_scaffold_file_path, mol)]
        final_all_sfcp_list = list(itertools.chain(*final_all_sfcp_list))

        # [5] Fragment identifying step #
        final_q_mol_fragment_list = [
            step_5.identify_Fragment_Smarts(self.all_scaffold_file_path, mol, self.final_all_fragments_Dic)]
        final_q_mol_fragment_list = list(itertools.chain(*final_q_mol_fragment_list))

        # merge NC-MFP info
        merge = final_all_scaffold_match_list + final_all_sfcp_list + final_q_mol_fragment_list

        final_bitstring = step_6.get_q_mol_nc_mfp_value_idx([merge], self.final_all_nc_mfp_labels)

        # reshape to have the same shape as the number of features
        final_bitstring = final_bitstring.reshape(final_bitstring.shape[1])
        return final_bitstring

    @staticmethod
    def generate_new_fingerprint_database(data: SmilesDataset, output_folder: str):
        """
      Generate a new fingerprint database from the given dataset.

      Parameters
      ----------
      data: SmilesDataset
        A dataset of SMILES strings.
      output_folder: str
        The output folder to save the database.
      
      Returns
      -------
      None
      
      """

        from .nc_mfp.generate_database import DatabaseGenerator

        # Generate the database
        DatabaseGenerator().generate_database(data, output_folder)
