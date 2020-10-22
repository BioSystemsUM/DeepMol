# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:36:08 2019

@author: jfsco
"""

import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, MACCSkeys, rdReducedGraphs, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

#TODO: redo comments
class fingerprintGenerator():
    """
    Class to generate fingerprints with RDKit

    Fix:
    Topological and RDK fingerprints giving the same results???? Maybe not
    """

    def __init__(self, dataset, smiles_label, class_label = None, fpt_type='morgan', to_dc=False):

        """
        Class to generate multiple RDKit fingerprint types.

        Parameters:
            -----------
            dataset: pandas dataframe with 'Smiles' and 'Class' columns
            fpt_type: name of the fingerprint to generate
                values: morgan --> Morgan Fingerprints (Circular Fingerprints)
                        topological --> Topological Fingerprints
                        rdkit --> RDKit fingerprints
                        layered --> Layered fingerprints
                        maccs --> MACCS Keys
                        atomPairs --> Atom Pairs Descriptors
                        topTorsions --> Topological Torsions
                        2DPharm --> 2D Pharmacophore Fingerprints
                        ergraphs --> Extended Reduced Graphs
        """

        if class_label is not None:
            self.dataset = pd.DataFrame({'Smiles': dataset[smiles_label].tolist(), 'Class': dataset[class_label].tolist()})
            self.labeled = True
        else:
            self.dataset = pd.DataFrame({'Smiles': dataset[smiles_label].tolist()})
            self.labeled = False

        self.fpt_type = fpt_type
        self.smiles_list = self.dataset.Smiles.tolist()
        if self.labeled:
            self.labels_list = self.dataset.Class.tolist()

    def _get_first_smile(self, smile):
        return smile.split(';')[0]

    def _convert_numpy_to_list(self, np_array):
        return np_array.tolist()

    def _convert_to_dc_dataset(self, X, y):
        return dc.data.DiskDataset.from_numpy(X=X, y=y)

    # Morgan Fingerprints (Circular Fingerprints)
    def _getMorganFingerprint(self, molecule, radius=2, nBits=1024):
        try:
            arr = np.zeros((1,))
            desc = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(molecule)), radius, nBits)
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # Topological Fingerprints
    def _getTopologicalFingerprint(self, molecule):
        try:
            arr = np.zeros((1,))
            desc = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(str(molecule)))
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # RDKit fingerprints
    def _rdkitFingerprint(self, molecule):
        try:
            arr = np.zeros((1,))
            desc = FingerprintMols.GetRDKFingerprint(Chem.MolFromSmiles(str(molecule)))
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # Layered fingerprints
    def _layeredFingerprint(self, molecule):
        try:
            arr = np.zeros((1,))
            desc = rdmolops.LayeredFingerprint(Chem.MolFromSmiles(str(molecule)))
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # MACCS Keys
    def _maccsKeys(self, molecule):
        try:
            arr = np.zeros((1,))
            desc = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(str(molecule)))
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # Atom Pairs Descriptors
    def _atomPairsDescriptors(self, molecule, nBits=2048):
        try:
            arr = np.zeros((1,))
            desc = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(Chem.MolFromSmiles(str(molecule)), nBits)
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # Topological Torsions
    def _topologicalTorsions(self, molecule, nBits=2048):
        try:
            arr = np.zeros((1,))
            desc = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(Chem.MolFromSmiles(str(molecule)),
                                                                                    nBits)
            DataStructs.ConvertToNumpyArray(desc, arr)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    # 2D Pharmacophore Fingerprints
    def _pharmacophore2DFingerprint(self, molecule):
        try:
            # arr = np.zeros((1,))
            desc = Generate.Gen2DFingerprint(Chem.MolFromSmiles(str(molecule)), Gobbi_Pharm2D.factory)
            arr = np.array(desc)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    #    # 3D Pharmacophore Fingerprints
    #    def _pharmacophore3DFingerprint(self, molecule):
    #        try:
    #            #arr = np.zeros((1,))
    #            mol = Chem.MolFromSmiles(str(molecule))
    #            desc = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory, dMat=Chem.Get3DDistanceMatrix(mol))
    #            arr = np.array(desc)
    #        except Exception as e:
    #            print('error in smile: ' + str(molecule))
    #            print(e)
    #            arr = np.nan
    #        return arr
    #

    # Extended Reduced Graphs
    # NOTE: these functions return an array of floats, not the usual fingerprint types
    def _extendedReducedGraphs(self, molecule):
        try:
            # arr = np.zeros((1,))
            desc = rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(str(molecule)))
            arr = np.array(desc)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def getFingerprintsDataset(self):

        dataset = self.dataset
        if self.fpt_type == 'morgan':
            dataset['ECFP'] = dataset['Smiles'].apply(self._getMorganFingerprint).tolist()
        elif self.fpt_type == 'topological':
            dataset['ECFP'] = dataset['Smiles'].apply(self._getTopologicalFingerprint).tolist()
        elif self.fpt_type == 'rdkit':
            dataset['ECFP'] = dataset['Smiles'].apply(self._rdkitFingerprint).tolist()
        elif self.fpt_type == 'layered':
            dataset['ECFP'] = dataset['Smiles'].apply(self._layeredFingerprint).tolist()
        elif self.fpt_type == 'maccs':
            dataset['ECFP'] = dataset['Smiles'].apply(self._maccsKeys).tolist()
        elif self.fpt_type == 'atomPairs':
            dataset['ECFP'] = dataset['Smiles'].apply(self._atomPairsDescriptors).tolist()
        elif self.fpt_type == 'topTorsions':
            dataset['ECFP'] = dataset['Smiles'].apply(self._topologicalTorsions).tolist()
        elif self.fpt_type == '2DPharm':
            dataset['ECFP'] = dataset['Smiles'].apply(self._pharmacophore2DFingerprint).tolist()
        #elif self.fpt_type == '3DPharm':
        #    dataset['ECFP'] = dataset['Smiles'].apply(self._pharmacophore3DFingerprint).tolist()
        elif self.fpt_type == 'ergraphs':
            dataset['ECFP'] = dataset['Smiles'].apply(self._extendedReducedGraphs).tolist()
        else:
            print('Invalid fingerprint name!')
            return pd.DataFrame()

        ecfp_df = dataset['ECFP'].apply(pd.Series)
        ecfp_df = ecfp_df.rename(columns=lambda x: 'FPT_' + str(x + 1))
        dataset = pd.concat([dataset, ecfp_df], axis=1).drop(['ECFP'], axis=1)
        print('Dataset dimensions: ', dataset.shape)
        #keep or remove dropna to maintain order??
        #TODO: check if works
        if self.to_dc:
            return self._convert_to_dc_dataset(ecfp_df, df['Smiles'])
        else:
            return dataset.dropna()


if __name__ == '__main__':
    df = pd.read_csv('dataset_last_version2.csv', sep=';', header=0)
    print(df.shape)
    mrg_fps = fingerprintGenerator(df, 'Smiles', 'Class', 'morgan')
    mrg_dataset = mrg_fps.getFingerprintsDataset()
    print(mrg_dataset.shape)
    # mrg_dataset.to_csv('mrg_fps.csv', sep = ';', index=False)