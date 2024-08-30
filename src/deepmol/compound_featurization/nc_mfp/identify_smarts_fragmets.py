import itertools
from deepmol.compound_featurization.nc_mfp.Preprocessing_step import Preprocessing
from deepmol.compound_featurization.nc_mfp.Scaffold_matching_step import ScaffoldMatching
from deepmol.compound_featurization.nc_mfp.Fragment_list_generation_step import Fragment_List_Generation
from deepmol.compound_featurization.nc_mfp.SFCP_assigning_step import SFCPAssigning
from deepmol.compound_featurization.nc_mfp.Fragment_identifying_step import FragmentIdentifying
from deepmol.compound_featurization.nc_mfp.Fingerprint_representation_step import FingerprintRepresentation

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

# Read Query Mols data
Query_FilePath = ('Data/nc_mfp_database.csv')
#Query_FilePath = 'C:\\Users\\Seomyungwon\\NC-MFP\\Data\\QueryMols_NC_MFP_Algorithm_TestSet.txt'

# Read All Scaffolds data
All_Scaffold_FilePath = ('All_Optimized_Scaffold_List.txt')
#All_Scaffold_FilePath = 'C:\\Users\\Seomyungwon\\NC-MFP\\Data\\All_Optimized_Scaffold_List.txt'

# Write NC-MFP file
OutputFilePath = ('FilePath/OutPutFileName.txt')
OutputFileName = ('OutputFile name.txt')


# read pickle with mols
import pickle
with open('mols.pickle', 'rb') as f:
    qMols_Smarts = pickle.load(f)

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

Final_all_Fragment_List = []
Final_all_Fragment_Dic = []

# import final_sfcp
import pickle
with open('final_sfcp.pickle', 'rb') as f:
    Final_all_SFCP_List = pickle.load(f)

# import Final_all_Fragment_Dic from pickle
import pickle

with open('fragment_dic.pickle', 'rb') as f:
    Final_all_Fragment_Dic = pickle.load(f)

parallel_callback = Parallel(n_jobs=15, backend="multiprocessing", prefer="threads")
with tqdm_joblib(tqdm(desc="get SFCP", total=len(qMols_Smarts))):
    res = parallel_callback(
        delayed(generate_info)(All_Scaffold_FilePath, entry, Final_all_Fragment_Dic)
        for entry in qMols_Smarts)
    
Final_qMol_Fragment_List = list(itertools.chain(*res))

# save to pickle Final_qMol_Fragment_List
with open('final_fragment_list.pickle', 'wb') as f:
    pickle.dump(Final_qMol_Fragment_List, f)