import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset
#from mordred import Calculator, descriptors, TopoPSA,Weight, CarbonTypes,SLogP, MoeType
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.linalg import block_diag
from scipy import stats
from rdkit import RDLogger



def get_fingerprints_user(data, label ,bitSize_circular=2048, morgan_radius=2):
    
    index_not_convertable = []
    
    """ 
    Computes the Fingerprints from Molecules
    """
    # if label is string get colum number

    #Disable printing Warnings
    RDLogger.DisableLog('rdApp.*')  
    
    feature_matrix= pd.DataFrame(np.zeros((data.shape[0],bitSize_circular)), dtype=int) 
    
    
    for i in tqdm(range(data.shape[0])):
       try:
           feature_matrix.iloc[i,:] = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data.iloc[i, label]),morgan_radius,nBits=bitSize_circular)) 
       except:
           feature_matrix.iloc[i,:] = 0
           index_not_convertable.append(i)
    RDLogger.EnableLog('rdApp.*')  
    
    if len(index_not_convertable)> 0:
        print("\n",len(index_not_convertable), " Molecules could not be read.")  
    
    
    return feature_matrix, index_not_convertable

def get_fingerprints(data, bitSize_circular=2048, labels_default=None , labels_morgan=None, morgan_radius=2):
    
    """ Computes the Fingerprints from Molecules
    """
    feature_matrix= pd.DataFrame(np.zeros((data.shape[0],bitSize_circular)), dtype=int) 
    for i in tqdm(range(data.shape[0])):
       feature_matrix.iloc[i,:] = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data.smiles.iloc[i]),morgan_radius,nBits=bitSize_circular)) 


    return(feature_matrix)


class FPAutoencoder_Dataset(Dataset):
    def __init__(self,fingerprint, np, npl):
        self.len = fingerprint.shape[0]
        self.fingerprint=(torch.tensor(fingerprint.values, dtype=torch.float))
        self.np = torch.tensor(np, dtype = torch.float)
        self.npl = torch.tensor(npl, dtype = torch.float)
    def __getitem__(self, index):
        return self.fingerprint[index], self.np[index], self.npl[index]
    
    def __len__(self):
        return self.len



class GraphDataset(Dataset):
    def __init__(self,feature, adjacency, target_reg, target_clf ):
        
        self.len = len(adjacency)
        self.adjacency = [torch.tensor(adj, dtype=torch.float) for adj in adjacency]
        self.feature = [torch.tensor(feat.values, dtype=torch.float) for feat in feature]
        self.target_reg = torch.tensor(target_reg, dtype=torch.float)
        self.target_clf = torch.tensor(target_clf, dtype=torch.float)
    def __getitem__(self, index):
        return self.feature[index], self.adjacency[index], self.adjacency[index].shape[0], self.target_reg[index], self.target_clf[index]
    
    def __len__(self):
        return self.len
    
def graph_collate(batch):
    feat      = [item[0] for item in batch ]
    adj       = [item[1] for item in batch ]
    sep_list  = [item[2] for item in batch ]
    target_reg    = torch.stack([item[3] for item in batch ])
    target_clf = torch.stack([item[4] for item in batch ])
    
    adj  = torch.tensor(block_diag(*adj), dtype=(torch.float))
    feat = torch.cat(feat, dim=0)
    
    return [feat, adj, sep_list], target_reg, target_clf.unsqueeze(1)


def create_gcn_features(smiles):
    print( "\n Generating Graph Conv Features ... \n") 
    # get Rdkit Molecules
    mols=[Chem.MolFromSmiles(x) for x in smiles]
    #mols= mols+maskri_mol
    #atom type
    atom_dummy=pd.get_dummies([atom.GetAtomicNum() for x in mols for atom in x.GetAtoms() ])
    #degree
    degree=pd.get_dummies([atom.GetDegree() for x in mols for atom in x.GetAtoms()])
    degree.columns=[ "degree_"+str(i) for i in range(degree.shape[1])]
    
    
    #Get Hybridization
    hy=[atom.GetHybridization() for x in mols for atom in x.GetAtoms()]
    hybridization=pd.get_dummies(hy)
    
    hybridization.columns=[ "hybrid_"+str(i) for i in range(hybridization.shape[1])]
    
    #aromaticity
    aromaticity =pd.get_dummies([atom.GetIsAromatic() for x in mols for atom in x.GetAtoms()])
    aromaticity=aromaticity.drop([0], axis=1)
    aromaticity.columns= ["InAromatic"]
    
    #formal charge
    formal_charge=pd.get_dummies([atom.GetFormalCharge() for x in mols for atom in x.GetAtoms()])
    formal_charge.columns=[ "charge_"+str(i) for i in range(formal_charge.shape[1])]
    
    #formal charge
    implicit_valence=pd.get_dummies([atom.GetImplicitValence() for x in mols for atom in x.GetAtoms()])
    implicit_valence.columns=[ "implicit_valence_"+str(i) for i in range(implicit_valence.shape[1])]
    
    #GetNumRadicalElectrons(
    chirality=pd.get_dummies([atom.GetChiralTag () for x in mols for atom in x.GetAtoms()])
    chirality.columns=[ "chirality_"+str(i) for i in range(chirality.shape[1])]
    
    #get protons
    num_h=pd.get_dummies([atom.GetNumImplicitHs() for x in mols for atom in x.GetAtoms()])
    num_h.columns=[ "num_h_"+str(i) for i in range(num_h.shape[1])]
    
    #concatenat features
    
    atom_features=pd.concat([atom_dummy,degree,num_h,chirality, implicit_valence,formal_charge,aromaticity, hybridization],axis=1)
    
    #usse only atom type to predict
    #atom_features=atom_dummy
    
    #generate Adjacency and Feature Matrix
    adjs=[None]*len(mols)
    feat=[None]*len(mols)
    index=0
    for i in range(len(mols)):
        A = GetAdjacencyMatrix(mols[i])
        adjs[i]=norm_adj(A)
        
        feat[i]=atom_features.iloc[index:(index+A.shape[0]),:].reset_index(drop=True)
        index+=A.shape[0]

    return adjs, feat



class FPDataset(Dataset):
    def __init__(self,fingerprint, target_reg, target_clf ):
        
        self.len = fingerprint.shape[0]
        
        self.fingerprint=(torch.tensor(fingerprint.values, dtype=torch.float))
        self.target_reg = torch.tensor(target_reg, dtype=torch.float)
        self.target_clf = torch.tensor(target_clf, dtype=torch.float)
    
    def __getitem__(self, index):
        return self.fingerprint[index], self.target_reg[index], self.target_clf[index]
    
    def __len__(self):
        return self.len
 
    
 
# =============================================================================
# def comp_descriptors(smiles):
#     mols = [Chem.MolFromSmiles(smile) for smile in smiles ]
#     calc = Calculator()
#     calc.register(MoeType.EState_VSA(1))
#     calc.register(MoeType.EState_VSA(2))
#     calc.register(MoeType.EState_VSA(3))
#     calc.register(MoeType.EState_VSA(4))
#     calc.register(MoeType.EState_VSA(5))
#     calc.register(MoeType.EState_VSA(6))
#     calc.register(MoeType.EState_VSA(7))
#     calc.register(MoeType.EState_VSA(8))
#     calc.register(MoeType.EState_VSA(9))
#     calc.register(MoeType.EState_VSA(10))
#     calc.register(MoeType.EState_VSA(11))
#     
#     calc.register(MoeType.PEOE_VSA(1))
#     calc.register(MoeType.PEOE_VSA(2))
#     calc.register(MoeType.PEOE_VSA(3))
#     calc.register(MoeType.PEOE_VSA(4))
#     calc.register(MoeType.PEOE_VSA(5))
#     calc.register(MoeType.PEOE_VSA(6))
#     calc.register(MoeType.PEOE_VSA(7))
#     calc.register(MoeType.PEOE_VSA(8))
#     calc.register(MoeType.PEOE_VSA(9))
#     calc.register(MoeType.PEOE_VSA(10))
#     calc.register(MoeType.PEOE_VSA(11))
#     calc.register(MoeType.PEOE_VSA(12))
#     calc.register(MoeType.PEOE_VSA(13))
#     calc.register(MoeType.PEOE_VSA(14))
#     
#     calc.register(MoeType.SMR_VSA(1))
#     calc.register(MoeType.SMR_VSA(2))
#     calc.register(MoeType.SMR_VSA(3))
#     calc.register(MoeType.SMR_VSA(4))
#     calc.register(MoeType.SMR_VSA(5))
#     calc.register(MoeType.SMR_VSA(6))
#     calc.register(MoeType.SMR_VSA(7))
#     calc.register(MoeType.SMR_VSA(8))
#     calc.register(MoeType.SMR_VSA(9))
#     calc.register(MoeType.SMR_VSA(10))
#     
#     calc.register(MoeType.SlogP_VSA(1))
#     calc.register(MoeType.SlogP_VSA(2))
#     calc.register(MoeType.SlogP_VSA(3))
#     calc.register(MoeType.SlogP_VSA(4))
#     calc.register(MoeType.SlogP_VSA(5))
#     calc.register(MoeType.SlogP_VSA(6))
#     calc.register(MoeType.SlogP_VSA(7))
#     calc.register(MoeType.SlogP_VSA(8))
#     calc.register(MoeType.SlogP_VSA(9))
#     calc.register(MoeType.SlogP_VSA(10))
#     calc.register(MoeType.SlogP_VSA(11))
#     calc.register(MoeType.SlogP_VSA(12))
#     
#     
#     calc.register(TopoPSA.TopoPSA(no_only=False))
#     
#     return calc.pandas(mols)
# 
# =============================================================================


def calculate_sensitivity_specificity(y_test, y_pred_test):
    # Note: More parameters are defined than necessary. 
    # This would allow return of other measures other than sensitivity and specificity
    
    # Get true/false for whether a breach actually occurred
    actual_pos = y_test == 1
    actual_neg = y_test == 0
    
    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (y_pred_test == 1) & (actual_pos)
    false_pos = (y_pred_test == 1) & (actual_neg)
    true_neg = (y_pred_test == 0) & (actual_neg)
    false_neg = (y_pred_test == 0) & (actual_pos)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_test == y_test)
    
    # Calculate sensitivity and specificity
    sensitivity = np.sum(true_pos) / np.sum(actual_pos)
    specificity = np.sum(true_neg) / np.sum(actual_neg)
    
    return sensitivity, specificity, accuracy


def norm_adj(x):
    """Normalizes Adjacency Matrix
    

    Parameters
    ----------
    x : matrix
        adjacency matrix

    Returns
    -------
    normlized adjacency matrix

    """
    
    x_hat=x+np.eye(x.shape[0])
    D_inv=np.diag(np.array(np.sum(x_hat, axis=1))**(-0.5))

    return(np.matmul(np.matmul(D_inv,x_hat), D_inv))




def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
