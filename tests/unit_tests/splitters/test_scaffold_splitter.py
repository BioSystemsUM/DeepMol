import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles

from deepmol.splitters import ScaffoldSplitter
from unit_tests.splitters.test_splitters import TestSplitters


class TestScaffoldSplitter(TestSplitters):
    
    def test_split(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 4)
        self.assertEqual(len(test_dataset.smiles), 1)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset.smiles) / 2)

    def test_scaffold_splitter_larger_dataset(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.dataset_to_test)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))

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
        self.assertEqual(len(train_dataset.smiles), 3435)
        self.assertEqual(len(test_dataset.smiles), 859)

    def test_scaffold_splitter_larger_dataset_binary_classification(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.binary_dataset, frac_train=0.5)

        self.assertEqual(len(train_dataset.smiles), len(test_dataset.smiles))
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

        self.assertGreater(counter, len(train_dataset.smiles) / 2)

        train_dataset, valid_dataset, test_dataset = similarity_splitter.train_valid_test_split(self.binary_dataset,
                                                                                                frac_train=0.5,
                                                                                                frac_test=0.3,
                                                                                                frac_valid=0.2)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(valid_dataset.smiles))

        self.assertAlmostEqual(len(train_dataset.y[train_dataset.y == 1]), len(train_dataset.y[train_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(test_dataset.y[test_dataset.y == 1]), len(test_dataset.y[test_dataset.y == 0]),
                               delta=10)

        self.assertAlmostEqual(len(valid_dataset.y[valid_dataset.y == 1]), len(valid_dataset.y[valid_dataset.y == 0]),
                               delta=10)

    def test_scaffold_splitter_invalid_smiles(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)

    def test_scaffold_splitter_invalid_smiles_and_nan(self):
        similarity_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)
