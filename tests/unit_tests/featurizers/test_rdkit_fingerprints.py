import os
from copy import copy
from unittest import TestCase

from IPython.core.display import SVG
from rdkit.Chem import MolFromSmiles

from deepmol.compound_featurization import MorganFingerprint, \
    MACCSkeysFingerprint, \
    LayeredFingerprint, RDKFingerprint, AtomPairFingerprint
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase

from tests import TEST_DIR


class TestRDKitFingerprints(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        # test Atom Pair fingerprints (without NaN generation)
        dataset_rows_number = len(self.mock_dataset.mols)
        AtomPairFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        MorganFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        MACCSkeysFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        LayeredFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        RDKFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        AtomPairFingerprint(n_jobs=1).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        MorganFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        MACCSkeysFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        LayeredFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        RDKFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_morgan_fingerprint_draw_bit(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        morgan_fingerprint = MorganFingerprint()
        depiction = morgan_fingerprint.draw_bit(molecule, 1,
                                                file_path=os.path.join(TEST_DIR, "data",
                                                                       'test_morgan_fingerprint_draw_bit.svg'))

        self.assertTrue(isinstance(depiction, SVG))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg')))
        os.remove(os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg'))

    def test_morgan_fingerprints_draw_bits(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        morgan_fingerprint = MorganFingerprint()
        depiction = morgan_fingerprint.draw_bits(molecule, [1, 114, 227], file_path=os.path.join(TEST_DIR,
                                                                                                 "data",
                                                                                                 'test_morgan_fingerprint_draw_bit.svg'))
        self.assertTrue(isinstance(depiction, SVG))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg')))

        depiction = morgan_fingerprint.draw_bits(molecule, "ON", file_path=os.path.join(TEST_DIR,
                                                                                        "data",
                                                                                        'test_morgan_fingerprint_draw_bit.svg'))
        self.assertTrue(isinstance(depiction, SVG))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg')))

        depiction = morgan_fingerprint.draw_bits(molecule, 1, file_path=os.path.join(TEST_DIR,
                                                                                        "data",
                                                                                        'test_morgan_fingerprint_draw_bit.svg'))
        self.assertTrue(isinstance(depiction, SVG))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg')))
        os.remove(os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg'))

    def test_morgan_fingerprint_draw_bit_with_invalid_molecule(self):
        dataset = copy(self.mock_dataset_with_invalid)
        molecule = dataset.smiles[-1]
        molecule = MolFromSmiles(molecule)
        morgan_fingerprint = MorganFingerprint()
        self.assertRaises(ValueError, morgan_fingerprint.draw_bit, molecule, 1,
                          file_path=os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg'))

    def test_macckeys_fingerprint_draw_bit(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        morgan_fingerprint = MACCSkeysFingerprint()
        depiction = morgan_fingerprint.draw_bit(molecule, 1,
                                                file_path=os.path.join(TEST_DIR, "data",
                                                                       'test_macckeys_fingerprint_draw_bit.svg'))

        self.assertTrue(isinstance(depiction, SVG))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", 'test_macckeys_fingerprint_draw_bit.svg')))
        os.remove(os.path.join(TEST_DIR, "data", 'test_macckeys_fingerprint_draw_bit.svg'))
