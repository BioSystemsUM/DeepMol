# -*- coding: utf-8 -*-
"""
Created on Fri Sept  11 15:03:54 2020

@author: jfsco
"""


################################################

# OLD FILE -- ONLY FOR REFERENCE

# REMOVE LATER

#################################################


from molvs import Standardizer
from rdkit import Chem
import pandas as pd
import numpy as np

def standardizer(mol):
    molecule = Chem.MolFromSmiles(mol)
    try:
        s = Standardizer()
        molecule = s.standardize(molecule)
        molecule = Chem.MolToSmiles(molecule)
    except Exception as e:
        print('error while standardizing smile: ' + str(mol))
        molecule = mol
    return molecule

def canonalize(mol):
    try :
        molecule = Chem.CanonSmiles(mol)
    except Exception as e:
        print('error while canonalizing smile: ' + str(mol))
        molecule = mol
    return molecule

def same_mol(m1, m2):
    return m1==m2


def preprocess(path, smiles_header, sep=',', header=0, n=None, save=True, save_path='preprocessed_dataset.csv'):

    dataset = pd.read_csv(path, sep=sep, header=header)

    if n is not None:
        dataset = dataset.sample(n=n)

    print('Preprocessing started...')
    print('Dataset size: ', dataset.shape)

    n = dataset.shape[0]

    print('Standardizing SMILES...')
    dataset['Standardized_Smiles'] = dataset['Smiles'].apply(standardizer).tolist()

    print('Canonalizing SMILES...')
    dataset['Canonical_Smiles'] = dataset['Smiles'].apply(canonalize).tolist()

    #dataset['Canonical_Smiles_From_Standardized_Smiles'] = dataset['Standardized_Smiles'].apply(canonalize).tolist()

    dataset = dataset.drop_duplicates(subset='Smiles', keep="first")

    print(n-dataset.shape[0], ' repeated smiles removed!')
    if save:
        dataset.to_csv(save_path, sep=',', header=True)

    return dataset, save_path


if __name__ == '__main__':
    dataset = preprocess(path='data/dataset_last_version2.csv', smiles_header='Smiles', sep=';', header=0, n=None)
    print(dataset[['Smiles', 'Standardized_Smiles', 'Canonical_Smiles']])
