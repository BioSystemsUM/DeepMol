import os
from unittest import TestCase, skip

from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles

from deepmol.loaders.loaders import CSVLoader, SDFLoader
from deepmol.splitters.splitters import SimilaritySplitter, ScaffoldSplitter, ButinaSplitter

import numpy as np

from tests import TEST_DIR


class TestSplitters(TestCase):

    def setUp(self) -> None:
        dataset = os.path.join(TEST_DIR, "data", "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Standardized_Smiles',
                           labels_fields=['Class'])

        self.mini_dataset_to_test = loader.create_dataset()

        dataset = os.path.join(TEST_DIR, "data", "PC-3.csv")
        loader = CSVLoader(dataset,
                           smiles_field='smiles',
                           labels_fields=['pIC50'])

        self.dataset_to_test = loader.create_dataset()

        dataset = os.path.join(TEST_DIR, "data", "invalid_smiles_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Standardized_Smiles',
                           labels_fields=['Class'])

        self.invalid_smiles_dataset = loader.create_dataset()

        dataset = os.path.join(TEST_DIR, "data", "dataset_sweet_3d_balanced.sdf")
        loader = SDFLoader(dataset,
                           labels_fields=['_SWEET'])

        self.binary_dataset = loader.create_dataset()

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def test_similarity_splitter(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 4)
        self.assertEqual(len(test_dataset), 1)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 2, 1024) for x in train_dataset.mols]

        mol2 = MolFromSmiles(test_dataset.mols[0])
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)

    def test_similarity_splitter_larger_dataset(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.dataset_to_test)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 3435)
        self.assertEqual(len(test_dataset), 859)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 2, 1024) for x in train_dataset.mols]

        mol2 = MolFromSmiles(test_dataset.mols[0])
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)

    def test_similarity_splitter_larger_dataset_binary_classification(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.binary_dataset)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)

        train_dataset, valid_dataset, test_dataset = similarity_splitter.train_valid_test_split(self.binary_dataset,
                                                                                                frac_train=0.8,
                                                                                                frac_valid=0.1)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertGreater(len(train_dataset), len(valid_dataset))

        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(test_dataset.y[test_dataset.y == 1]), len(test_dataset.y[test_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(valid_dataset.y[valid_dataset.y == 1]), len(valid_dataset.y[valid_dataset.y == 0]),
                               delta=10)

    def test_similarity_splitter_invalid_smiles(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(test_dataset), 1)

    def test_similarity_splitter_invalid_smiles_and_nan(self):

        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(test_dataset), 1)

    def test_scaffold_splitter(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 4)
        self.assertEqual(len(test_dataset), 1)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 2, 1024) for x in train_dataset.mols]

        mol2 = MolFromSmiles(test_dataset.mols[0])
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)

    def test_scaffold_splitter_larger_dataset(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.dataset_to_test)
        self.assertGreater(len(train_dataset), len(test_dataset))

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 2, 1024) for x in train_dataset.mols]

        mol2 = MolFromSmiles(test_dataset.mols[0])
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)
        self.assertEqual(len(train_dataset), 3435)
        self.assertEqual(len(test_dataset), 859)

    def test_scaffold_splitter_larger_dataset_binary_classification(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.binary_dataset, frac_train=0.5)

        self.assertEqual(len(train_dataset), len(test_dataset))
        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(test_dataset.y[test_dataset.y == 1]), len(test_dataset.y[test_dataset.y == 0]),
                               delta=10)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)

        train_dataset, valid_dataset, test_dataset = similarity_splitter.train_valid_test_split(self.binary_dataset,
                                                                                                frac_train=0.5,
                                                                                                frac_test=0.3,
                                                                                                frac_valid=0.2)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertGreater(len(train_dataset), len(valid_dataset))

        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(test_dataset.y[test_dataset.y == 1]), len(test_dataset.y[test_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(valid_dataset.y[valid_dataset.y == 1]), len(valid_dataset.y[valid_dataset.y == 0]),
                               delta=10)

    def test_scaffold_splitter_invalid_smiles(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(test_dataset), 1)

    def test_scaffold_splitter_invalid_smiles_and_nan(self):

        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(test_dataset), 1)

    def test_butina_splitter(self):
        similarity_splitter = ButinaSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 5)
        self.assertEqual(len(test_dataset), 0)

    def test_butina_splitter_larger_dataset(self):
        similarity_splitter = ButinaSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.dataset_to_test,
                                                                           homogenous_datasets=True)
        self.assertGreater(len(train_dataset), len(test_dataset))

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), 2, 1024) for x in train_dataset.mols]

        mol2 = MolFromSmiles(test_dataset.mols[0])
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)
        self.assertEqual(len(train_dataset), 3435)
        self.assertEqual(len(test_dataset), 859)

    def test_butina_splitter_invalid_smiles(self):
        similarity_splitter = ButinaSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(test_dataset), 1)

    def test_butina_splitter_invalid_smiles_and_nan(self):

        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        similarity_splitter = ButinaSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(test_dataset), 1)

    def test_butina_splitter_larger_dataset_binary_classification(self):

        similarity_splitter = ButinaSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.binary_dataset, frac_train=0.7)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(test_dataset.y[test_dataset.y == 1]), len(test_dataset.y[test_dataset.y == 0]),
                               delta=10)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset) / 2)

        train_dataset, valid_dataset, test_dataset = similarity_splitter.train_valid_test_split(self.binary_dataset,
                                                                                                frac_train=0.5,
                                                                                                frac_test=0.3,
                                                                                                frac_valid=0.2)

        self.assertGreater(len(train_dataset), len(test_dataset))
        self.assertGreater(len(train_dataset), len(valid_dataset))

        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(test_dataset.y[test_dataset.y == 1]), len(test_dataset.y[test_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(valid_dataset.y[valid_dataset.y == 1]), len(valid_dataset.y[valid_dataset.y == 0]),
                               delta=10)
