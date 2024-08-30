from collections import OrderedDict
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import joblib
import contextlib

import numpy as np

def get_string(Final_all_NC_MFP_Info, Final_all_NC_MFP_Label, Info):
    #Initialize
    Final_qMol_NC_MFP_Value = np.zeros((1, len(Final_all_NC_MFP_Label)))
    for ai in range(0, len(Final_all_NC_MFP_Label)):

        if Final_all_NC_MFP_Info[Info].__contains__(Final_all_NC_MFP_Label[ai]):
            Final_qMol_NC_MFP_Value[0, ai] = 1
        else:
            Final_qMol_NC_MFP_Value[0, ai] = 0

    # for bi in range(0, len(Final_qMol_NC_MFP_Value)):
    #     Final_qMOl_NC_MFP_Value_String = Final_qMOl_NC_MFP_Value_String + str(Final_qMol_NC_MFP_Value[bi]) + "\t"
    # Final_qMOl_NC_MFP_Value_String  = Final_qMOl_NC_MFP_Value_String + str('QueryMol_')+ str(Info+1) + "\n"

    return Final_qMol_NC_MFP_Value

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



class FingerprintRepresentation:

    def get_all_NC_MFP_Label(self, Final_all_NC_MFP_Info):

        Final_all_NC_MFP_Label = []

        # Final_NC_MFP_BitString = list(itertools.chain(*Final_all_NC_MFP_Info))
        Final_NC_MFP_BitString = Final_all_NC_MFP_Info

        for ai in range (0, len(Final_NC_MFP_BitString)):
            Final_all_NC_MFP_Label.append(Final_NC_MFP_BitString[ai])

        Final_all_NC_MFP_Label = list(OrderedDict.fromkeys(Final_all_NC_MFP_Label))
        Final_all_NC_MFP_Label.sort()

        return Final_all_NC_MFP_Label


    def get_qMol_NC_MFP_Value(self, Final_all_NC_MFP_Info, Final_all_NC_MFP_Label):

        Final_qMol_NC_MFP_Value = []
        Final_qMOl_NC_MFP_Value_String = ""

        for i in range (0, len(Final_all_NC_MFP_Label)):
            Final_qMOl_NC_MFP_Value_String = Final_qMOl_NC_MFP_Value_String  + str(Final_all_NC_MFP_Label[i]) + "\t"
        Final_qMOl_NC_MFP_Value_String  = Final_qMOl_NC_MFP_Value_String + "\n"

        for qMol_NC_MFP_Info in Final_all_NC_MFP_Info:
            for ai in range(0, len(Final_all_NC_MFP_Label)):

                if qMol_NC_MFP_Info.__contains__(Final_all_NC_MFP_Label[ai]):
                    Final_qMol_NC_MFP_Value.append('1')
                else:
                    Final_qMol_NC_MFP_Value.append('0')

            for bi in range(0, len(Final_qMol_NC_MFP_Value)):
                Final_qMOl_NC_MFP_Value_String = Final_qMOl_NC_MFP_Value_String + str(Final_qMol_NC_MFP_Value[bi]) + "\t"
            Final_qMOl_NC_MFP_Value_String  = Final_qMOl_NC_MFP_Value_String + "\n"

            #Initialize
            Final_qMol_NC_MFP_Value = []

        return Final_qMOl_NC_MFP_Value_String

    def represent_NC_MFP(self, OutputFilePath, OutputFileName, Final_BitString):

        file = open(OutputFilePath + OutputFileName, 'w')

        file.write(Final_BitString)
        file.close()


    def get_q_mol_nc_mfp_value_idx(self, Final_all_NC_MFP_Info, Final_all_NC_MFP_Label):

        Final_qMOl_NC_MFP_Value_String = ""

        # add a tqdm 
        # from tqdm import tqdm
        # for i in tqdm(range (0, len(Final_all_NC_MFP_Label))):
        #     Final_qMOl_NC_MFP_Value_String = Final_qMOl_NC_MFP_Value_String + str(Final_all_NC_MFP_Label[i]) + "\t"
        # Final_qMOl_NC_MFP_Value_String  = Final_qMOl_NC_MFP_Value_String + str('NC-MFP_BitString') + "\n"


        Final_qMOl_NC_MFP_Value_String = get_string(Final_all_NC_MFP_Info, Final_all_NC_MFP_Label, 0)

        # Final_qMOl_NC_MFP_Value_String = list(itertools.chain(*res))
        Final_qMOl_NC_MFP_Value_String = np.array(Final_qMOl_NC_MFP_Value_String)

        return Final_qMOl_NC_MFP_Value_String