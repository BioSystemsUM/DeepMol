from deepmol.compound_featurization import MorganFingerprint, LayeredFingerprint, TwoDimensionDescriptors
from deepmol.compound_featurization.rdkit_descriptors import ThreeDimensionalMoleculeGenerator
from tests.integration_tests.dataset.test_dataset import TestDataset
from deepmol.compound_featurization import MHFP, BiosynfoniKeys

class TestDatasetFeaturizers(TestDataset):

    def test_clear_cached_property_after_generating_features(self):
        MorganFingerprint().featurize(self.small_dataset_to_test, inplace=True)

        self.assertEqual(self.small_dataset_to_test.X.shape, (13, 2048))

        LayeredFingerprint(fpSize=1024).featurize(self.small_dataset_to_test, inplace=True)

        self.assertEqual(self.small_dataset_to_test.X.shape, (13, 1024))

        TwoDimensionDescriptors().featurize(self.small_dataset_to_test, inplace=True)

    def test_dataset_with_similarity_matrix(self):
        from deepmol.compound_featurization import TanimotoSimilarityMatrix
        dataset_rows_number = len(self.small_dataset_to_test.mols)
        TanimotoSimilarityMatrix(n_molecules=4).featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(dataset_rows_number, self.small_dataset_to_test._X.shape[0])
        self.assertEqual(4, self.small_dataset_to_test._X.shape[1])

    def test_dataset_with_nc_fp(self):
        from deepmol.compound_featurization import NcMfp
        NcMfp().featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(self.small_dataset_to_test.X.shape[0], 13)
        self.assertEqual(self.small_dataset_to_test.X.shape[1], 254399)

    def test_dataset_with_neural_npfp(self):
        from deepmol.compound_featurization import NeuralNPFP
        NeuralNPFP().featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(self.small_dataset_to_test.X.shape[0], 13)
        self.assertEqual(self.small_dataset_to_test.X.shape[1], 64)

    def test_mhfp_featurizer(self):
        MHFP().featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(self.small_dataset_to_test.X.shape[0], 13)
        self.assertEqual(self.small_dataset_to_test.X.shape[1], 2048)

    def test_biosynfoni_featurizer(self):
        BiosynfoniKeys().featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(self.small_dataset_to_test.X.shape[0], 13)
        self.assertEqual(self.small_dataset_to_test.X.shape[1], 39)

    def test_generate_structures(self):
        generator = ThreeDimensionalMoleculeGenerator()
        generator.generate(self.small_dataset_to_test)

        for mol in self.small_dataset_to_test.mols:
            self.assertGreater(len(mol.GetConformers()), 0)
