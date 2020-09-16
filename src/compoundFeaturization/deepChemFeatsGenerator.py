# -*- coding: utf-8 -*-
"""
Created on Wed Sep  10 16:16:54 2020

@author: jfsco
"""

import pandas as pd
import numpy as np
import rdkit
import deepchem as dc
from deepchem.utils import conformers
from rdkit import Chem
#from deepchem.feat import RDKitDescriptors, BPSymmetryFunctionInput, CoulombMatrix, CoulombMatrixEig


class DeepChemFeaturizerGenerator():
    """
    ...

    """

    def __init__(self, dataset, smiles_label, class_label=None, fpt_type='morgan'):

        """

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


    def _RDKitDescriptors(self, molecule):
        '''
        RDKitDescriptors featurizes a molecule by computing descriptors values for specified descriptors. Intrinsic to
        the featurizer is a set of allowed descriptors, which can be accessed using RDKitDescriptors.allowedDescriptors.
        '''
        try:
            featurizer = dc.feat.RDKitDescriptors()
            features = featurizer.featurize(molecule)
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _BPSymmetryFunction(self, molecule): #TODO: not working properly
        '''
        Behler-Parinello Symmetry function or BPSymmetryFunction featurizes a molecule by computing the atomic
        number and coordinates for each atom in the molecule.
        '''
        try:
            #engine = conformers.ConformerGenerator(max_conformers=1)
            #mol = engine.generate_conformers(molecule)

            featurizer = dc.feat.BPSymmetryFunctionInput(max_atoms=20)
            features = featurizer.featurize(mol=molecule)

            arr = np.array(features)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr


    def _CoulombMatrix(self, molecule): #TODO: not working properly
        '''
        CoulombMatrix, featurizes a molecule by computing the coulomb matrices for different conformers
        of the molecule, and returning it as a list.

        A Coulomb matrix tries to encode the energy structure of a molecule. The matrix is symmetric, with the
        off-diagonal elements capturing the Coulombic repulsion between pairs of atoms and the diagonal elements
        capturing atomic energies using the atomic numbers.
        '''
        try:
            #molecule = Chem.MolFromSmiles(molecule)
            #engine = conformers.ConformerGenerator(max_conformers=1)
            #mol = engine.generate_conformers(molecule)

            featurizer = dc.feat.CoulombMatrix(max_atoms=20, randomize=False, remove_hydrogens=False, upper_tri=False, n_samples=1, seed=None)
            features = featurizer.featurize(molecule)
            print(features)
            arr = np.array(features[0])
        except Exception as e:
            print(e)
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _CoulombMatrixEig(self, molecule): #TODO: not working properly
        '''
        CoulombMatrix is invariant to molecular rotation and translation, since the interatomic distances or atomic
        numbers do not change. However the matrix is not invariant to random permutations of the atom's indices.
        To deal with this, the CoulumbMatrixEig featurizer was introduced, which uses the eigenvalue spectrum of the
        columb matrix, and is invariant to random permutations of the atom's indices.
        '''
        try:
            #engine = conformers.ConformerGenerator(max_conformers=1)
            #mol = engine.generate_conformers(molecule)

            featurizer = dc.feat.CoulombMatrixEig(max_atoms=20, randomize=False, remove_hydrogens=False, n_samples=1, seed=None)
            features = featurizer.featurize(molecule)
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _ConvMolFeaturizer(self, molecule):
        '''

        '''
        try:
            featurizer = dc.feat.ConvMolFeaturizer(master_atom=False, use_chirality=False, atom_properties=[])
            features = featurizer.featurize(molecule)
            #TODO: append features[0] or features[0].atom_features
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _WeaveFeaturizer(self, molecule):
        '''

        return a WeaveMol object --> .nodes, .pairs, .num_atoms, .n_features, .pair_edges $TODO: what should i return?
        '''
        try:
            featurizer = dc.feat.WeaveFeaturizer(graph_distance=True, explicit_H=False, use_chirality=False, max_pair_distance=None)
            features = featurizer.featurize(molecule)

            arr = np.array(features)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr


    def _CircularFingerprint(self, molecule):
        '''

        '''
        try:
            featurizer = dc.feat.CircularFingerprint(radius=2, size=2048, chiral=False, bonds=True, features=False, sparse=False, smiles=False)
            features = featurizer.featurize(molecule)
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _Mol2VecFingerprint(self, molecule):
        '''

        '''
        try:
            featurizer = dc.feat.Mol2VecFingerprint(pretrain_model_path=None, radius=1, unseen='UNK', gather_method='sum')
            features = featurizer.featurize(molecule)
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _MordredDescriptors(self, molecule): #TODO: install mordred and test again
        '''

        '''
        try:
            featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
            features = featurizer.featurize(molecule)
            arr = np.array(features)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _AtomCoordinates(self, molecule): #TODO: not working properly
        '''

        '''
        try:
            featurizer = dc.feat.AtomicCoordinates()
            features = featurizer.featurize(molecule)
            arr = np.array(features[0])
        except Exception as e:
            print(e)
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    #TODO: Define char_to_idx !!!!
    def _SmilesToSeq(self, molecule):
        '''
        remove_pad(characters: List[str]) → List[str]
        Removes PAD_TOKEN from the character list.

        smiles_from_seq(seq: List[int]) → str
        Reconstructs SMILES string from sequence.

        to_seq(smile: List[str]) → numpy.ndarray
        Turns list of smiles characters into array of indices
        '''
        try:
            #featurizer = dc.feat.SmilesToSeq(char_to_idx: Dict[str, int], max_len=250, pad_len=10)
            features = featurizer.featurize(molecule)
            arr = np.array(features)
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _SmilesToImage(self, molecule): #TODO: solve image array problem while applying dataset['ECFP'].apply(pd.Series)
        '''

        '''
        try:
            featurizer = dc.feat.SmilesToImage(img_size=80, res=0.5, max_len=250, img_spec='std')
            features = featurizer.featurize(molecule)
            print(features[0])
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr

    def _OneHotFeaturizer(self, molecule): #TODO: solve onehotencoding array problem while applying dataset['ECFP'].apply(pd.Series)
        '''
        pad_smile(smiles: str) → str
        Pad SMILES string to self.pad_length

        Parameters:	smiles (str) – The smiles string to be padded.
        Returns:	SMILES string space padded to self.pad_length
        Return type:	str

        untransform(one_hot_vectors: numpy.ndarray) → str
        Convert from one hot representation back to SMILES

        Parameters:	one_hot_vectors (np.ndarray) – An array of one hot encoded features.
        Returns:	SMILES string for an one hot encoded array.
        Return type:	str
        '''
        try:
            featurizer = dc.feat.OneHotFeaturizer(charset=['#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 'p', 's', 'r'], max_length=100)
            features = featurizer.featurize(molecule)
            arr = np.array(features[0])
        except Exception as e:
            print('error in smile: ' + str(molecule))
            arr = np.nan
        return arr


    def getFeaturizerDataset(self):

        dataset = self.dataset
        if self.fpt_type == 'rdkit':
            dataset['ECFP'] = dataset['Smiles'].apply(self._RDKitDescriptors).tolist()
        elif self.fpt_type == 'bps':
            dataset['ECFP'] = dataset['Smiles'].apply(self._BPSymmetryFunction).tolist()
        elif self.fpt_type == 'coulomb':
            dataset['ECFP'] = dataset['Smiles'].apply(self._CoulombMatrix).tolist()
        elif self.fpt_type == 'coulombEig':
            dataset['ECFP'] = dataset['Smiles'].apply(self._CoulombMatrixEig).tolist()
        elif self.fpt_type == 'convMol':
            dataset['ECFP'] = dataset['Smiles'].apply(self._ConvMolFeaturizer).tolist()
        elif self.fpt_type == 'weave':
            dataset['ECFP'] = dataset['Smiles'].apply(self._WeaveFeaturizer).tolist()
        elif self.fpt_type == 'circular':
            dataset['ECFP'] = dataset['Smiles'].apply(self._CircularFingerprint).tolist()
        elif self.fpt_type == 'mol2vec':
            dataset['ECFP'] = dataset['Smiles'].apply(self._Mol2VecFingerprint).tolist()
        elif self.fpt_type == 'mordred':
            dataset['ECFP'] = dataset['Smiles'].apply(self._MordredDescriptors).tolist()
        elif self.fpt_type == 'atomcoord':
            dataset['ECFP'] = dataset['Smiles'].apply(self._AtomCoordinates).tolist()
        elif self.fpt_type == 'smiles2seq':
            dataset['ECFP'] = dataset['Smiles'].apply(self._SmilesToSeq).tolist()
        elif self.fpt_type == 'smiles2image':
            dataset['ECFP'] = dataset['Smiles'].apply(self._SmilesToImage).tolist()
        elif self.fpt_type == 'onehot':
            dataset['ECFP'] = dataset['Smiles'].apply(self._OneHotFeaturizer).tolist()
        else:
            print('Invalid featurizer name!')
            return pd.DataFrame()

        ecfp_df = dataset['ECFP'].apply(pd.Series)
        ecfp_df = ecfp_df.rename(columns=lambda x: 'Feat_' + str(x + 1))
        dataset = pd.concat([dataset, ecfp_df], axis=1).drop(['ECFP'], axis=1)
        print('Dataset dimensions: ', dataset.shape)
        #keep or remove dropna to maintain order??
        return dataset.dropna()

if __name__ == '__main__':
    df = pd.read_csv('dataset_last_version2.csv', sep=';', header=0)[:10]
    print(df.shape)
    rdkit_fps = DeepChemFeaturizerGenerator(df, 'Smiles', 'Class', 'rdkit')
    rdkit_dataset = rdkit_fps.getFeaturizerDataset()
    print(rdkit_dataset.shape)
    print(rdkit_dataset)