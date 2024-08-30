import itertools
import os

from deepmol.datasets.datasets import Dataset
from .Preprocessing_step import Preprocessing
from .Scaffold_matching_step import ScaffoldMatching
from .Fragment_list_generation_step import Fragment_List_Generation
from .SFCP_assigning_step import SFCPAssigning
from .Fragment_identifying_step import FragmentIdentifying
from .Fingerprint_representation_step import FingerprintRepresentation

# coding: utf-8
# Input: Smarts of query compound (.txt)
# Output: NC-MFP bit string (.txt)

# NC-MFP calculation algorithm #
# An Example set for the NC-MFP calculation: 20 query compounds in NPASS DB
# ('Data/QueryMols_NC_MFP_Algorithm_TestSet.txt')

# Define classes
Step_1 = Preprocessing()
Step_2 = ScaffoldMatching()
Step_3 = Fragment_List_Generation()
Step_4 = SFCPAssigning()
Step_5 = FragmentIdentifying()
Step_6 = FingerprintRepresentation()

# Read All Scaffolds data
this_file_directory = os.path.dirname(os.path.abspath(__file__))
All_Scaffold_FilePath = (os.path.join(this_file_directory, 'data/All_Optimized_Scaffold_List.txt'))
#All_Scaffold_FilePath = 'C:\\Users\\Seomyungwon\\NC-MFP\\Data\\All_Optimized_Scaffold_List.txt'

from joblib import Parallel, delayed
from tqdm import tqdm
import joblib
import contextlib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def generate_info(All_Scaffold_FilePath, entry, Final_all_Fragment_Dic):
    try:
        # [2] Scaffold matching step #
        Final_all_Scaffold_Match_List = []
        Final_all_Scaffold_Match_List.append(Step_2.match_all_scaffold_smarts(All_Scaffold_FilePath, entry))

        # Merge scaffold lists
        Final_all_Scaffold_Match_List = list(itertools.chain(*Final_all_Scaffold_Match_List))
        Final_all_Scaffold_Match_List = list(set(Final_all_Scaffold_Match_List))

    except IndexError as e:
        print(e)

    try:
        # [3] Fragment list generation & [4] Scfffold-Fragment Connection List(SFCP) assigning step #
        Final_all_SFCP_List = []
        Final_all_SFCP_List.append(Step_4.assign_All_SFCP_Smarts(All_Scaffold_FilePath, entry))
        Final_all_SFCP_List = list(itertools.chain(*Final_all_SFCP_List))

    except IndexError as e:
        print(e)

    try:
        # [5] Fragment identifying step #
        Final_qMol_Fragment_List = []
        Final_qMol_Fragment_List.append(Step_5.identify_Fragment_Smarts(All_Scaffold_FilePath, entry, Final_all_Fragment_Dic))
        Final_qMol_Fragment_List = list(itertools.chain(*Final_qMol_Fragment_List))

        # merge NC-MFP info
        merge = Final_all_Scaffold_Match_List + Final_all_SFCP_List + Final_qMol_Fragment_List

        return merge

    except IndexError as e:
        print(e)

class DatabaseGenerator():
    def __init__(self):
        self.All_Scaffold_FilePath = All_Scaffold_FilePath

    def generate_database(self, dataset: Dataset, output_folder: str):

        # All fragments generation #
        final_all_fragment_list = []

        for qMol_Smart in dataset.smiles:
            final_all_fragment_list.append(Step_3.generate_all_Fragment_List(All_Scaffold_FilePath, qMol_Smart))

        # merge list
        final_all_fragment_list = list(itertools.chain(*final_all_fragment_list))
        # remove duplicated elements in list
        final_all_fragment_list = list(set(final_all_fragment_list))

        final_all_fragment_dic = Step_3.generate_all_Fragment_Dictionary(final_all_fragment_list)

        parallel_callback = Parallel(n_jobs=15, backend="multiprocessing", prefer="threads")
        with tqdm_joblib(tqdm(desc="generate info", total=len(dataset.smiles))):
            res = parallel_callback(
                delayed(generate_info)(All_Scaffold_FilePath, entry, final_all_fragment_dic)
                for entry in dataset.smiles)
            
        final_all_NC_MFP_info = list(itertools.chain(*res))

        import pickle

        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, 'fragment_dic.pickle'), 'wb') as f:
            pickle.dump(final_all_fragment_dic, f)


        final_label = Step_6.get_all_NC_MFP_Label(final_all_NC_MFP_info)

        with open(os.path.join(output_folder, 'final_label.pickle'), 'wb') as f:
            pickle.dump(final_label, f)

        

        




