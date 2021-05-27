import numpy as np
import pandas as pd
from rdkit import Chem


df = pd.read_excel('data/DTP_NCI60_RAW.xlsx', sheet_name='all', skiprows=8, engine='openpyxl',
                   usecols=lambda x: x not in ['FDA status', 'Mechanism of action', 'Total probes', 'Experiment name', 'PubChem SID'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.replace('na', np.nan, inplace=True)
df = df[df['Failure reason'] == '-']
#df.isna().sum().to_csv('na_counts.csv')
df2 = df[['NSC #', 'LC:A549/ATCC']]
df2.dropna(axis=0, inplace=True)

df2['NSC #'] = df['NSC #'].astype(int)
df2['LC:A549/ATCC'] = df['LC:A549/ATCC'].astype(float)
df2['NSC #'] = df['NSC #'].astype(int).astype(str)
df_avg = df2.groupby(['NSC #'], as_index=False).mean()  # calculate average -log(GI50) as there are compounds with multiple replicates

# Get SMILES strings for compounds
nsc_to_smiles = {'NSC #': [],
                 'smiles': []}
sppl = Chem.SDMolSupplier('data/Chem2D_Jun2016.sdf')
for mol in sppl:
    if mol is not None:# some compounds cannot be loaded
        nsc_to_smiles['NSC #'].append(str(mol.GetProp('NSC')))
        nsc_to_smiles['smiles'].append(Chem.MolToSmiles(mol))
nsc_to_smiles_df = pd.DataFrame(data=nsc_to_smiles)

df_avg = df_avg.merge(nsc_to_smiles_df, how='left', on='NSC #')
df_avg.dropna(axis=0, inplace=True) # 20730 rows
df_avg.to_csv('data/nci60_a549atcc_gi50.csv', index=False)