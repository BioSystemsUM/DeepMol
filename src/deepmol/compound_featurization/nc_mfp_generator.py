import pickle
from deepmol.compound_featurization.base_featurizer import MolecularFeaturizer
from deepmol.datasets.datasets import SmilesDataset
from .nc_mfp.Preprocessing_step import Preprocessing
from .nc_mfp.Scaffold_matching_step import Scaffold_Matching
from .nc_mfp.Fragment_list_generation_step import Fragment_List_Generation
from .nc_mfp.SFCP_assigning_step import SFCP_Assigning
from .nc_mfp.Fragment_identifying_step import Fragment_identifying
from .nc_mfp.Fingerprint_representation_step import Fingerprint_representation
from .nc_mfp import NC_MFP_PATH

import os

import numpy as np
from rdkit.Chem import Mol, MolToSmiles
import os

import pickle
import itertools

class NC_MFP(MolecularFeaturizer):

    def __init__(self, database_file_path = os.path.join(NC_MFP_PATH, 'databases', 'ncdb'), **kwargs) -> None:
        super().__init__(**kwargs)

        self.all_scaffold_file_path = (os.path.join(NC_MFP_PATH, "databases", 'All_Optimized_Scaffold_List.txt'))
        # import Final_all_Fragment_Dic from pickle
        with open(os.path.join(database_file_path, 'fragment_dic.pickle'), 'rb') as f:
            
            self.final_all_fragments_Dic = pickle.load(f)

        with open(os.path.join(database_file_path, 'final_label.pickle'), 'rb') as f:
            self.final_all_NC_MFP_labels = pickle.load(f)
        
        self.feature_names = [f'nc_mfp_{i}' for i in range(len(self.final_all_NC_MFP_labels))]

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
        Step_2 = Scaffold_Matching()
        Step_4 = SFCP_Assigning()
        Step_5 = Fragment_identifying()
        Step_6 = Fingerprint_representation()

        mol = MolToSmiles(mol)

        # [2] Scaffold matching step #
        final_all_scaffold_match_list = []
        final_all_scaffold_match_list.append(Step_2.match_All_Scaffold_Smarts(self.all_scaffold_file_path, mol))

        # Merge scaffold lists
        final_all_scaffold_match_list = list(itertools.chain(*final_all_scaffold_match_list))
        final_all_scaffold_match_list = list(set(final_all_scaffold_match_list))

        # [3] Fragment list generation & [4] Scfffold-Fragment Connection List(SFCP) assigning step #
        final_all_SFCP_list = []
        final_all_SFCP_list.append(Step_4.assign_All_SFCP_Smarts(self.all_scaffold_file_path, mol))
        final_all_SFCP_list = list(itertools.chain(*final_all_SFCP_list))


        # [5] Fragment identifying step #
        Final_qMol_Fragment_List = []
        Final_qMol_Fragment_List.append(Step_5.identify_Fragment_Smarts(self.all_scaffold_file_path, mol, self.final_all_fragments_Dic))
        Final_qMol_Fragment_List = list(itertools.chain(*Final_qMol_Fragment_List))

        # merge NC-MFP info
        merge = final_all_scaffold_match_list + final_all_SFCP_list + Final_qMol_Fragment_List

        final_bitstring = Step_6.get_qMol_NC_MFP_Value_Idx([merge], self.final_all_NC_MFP_labels)

        #reshape to have the same shape as the number of features
        final_bitstring = final_bitstring.reshape(final_bitstring.shape[1])
        return final_bitstring

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