import os
import shutil
import sys
from copy import copy
from unittest import TestCase

from IPython.core.display import SVG
from PIL.PngImagePlugin import PngImageFile
from rdkit.Chem import MolFromSmiles

from deepmol.compound_featurization import MorganFingerprint, \
    MACCSkeysFingerprint, \
    LayeredFingerprint, RDKFingerprint, AtomPairFingerprint
from deepmol.compound_featurization.rdkit_fingerprints import AtomPairFingerprintCallbackHash
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

        AtomPairFingerprintCallbackHash().featurize(self.mock_dataset)
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

        dataset = copy(self.mock_dataset_with_invalid)
        AtomPairFingerprintCallbackHash().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])



    def test_units_of_fingerprints(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        fp = MorganFingerprint()._featurize(molecule)
        self.assertEqual(fp.shape[0], 2048)
        fp = AtomPairFingerprint()._featurize(molecule)
        self.assertEqual(fp.shape[0], 2048)
        fp = MACCSkeysFingerprint()._featurize(molecule)
        self.assertEqual(fp.shape[0], 167)
        fp = LayeredFingerprint()._featurize(molecule)
        self.assertEqual(fp.shape[0], 2048)
        fp = RDKFingerprint()._featurize(molecule)
        self.assertEqual(fp.shape[0], 2048)
        fp = AtomPairFingerprintCallbackHash()._featurize(molecule)
        self.assertEqual(fp.shape[0], 2048)

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

        self.assertRaises(ValueError, morgan_fingerprint.draw_bit, molecule, 0)

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

    def test_morgan_fingerprint_draw_bits_to_fail(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        morgan_fingerprint = MorganFingerprint()
        self.assertRaises(ValueError, morgan_fingerprint.draw_bits, molecule, 0)
        morgan_fingerprint.draw_bits(molecule, [1, 2, 3])
        self.assertRaises(ValueError, morgan_fingerprint.draw_bits, molecule, [13, 20, 3])
        self.assertRaises(ValueError, morgan_fingerprint.draw_bits, molecule, "OFF")

    def test_morgan_fingerprint_draw_bit_with_invalid_molecule(self):
        dataset = copy(self.mock_dataset_with_invalid)
        molecule = dataset.smiles[-1]
        molecule = MolFromSmiles(molecule)
        morgan_fingerprint = MorganFingerprint()
        self.assertRaises(ValueError, morgan_fingerprint.draw_bit, molecule, 1,
                          file_path=os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg'))
        self.assertRaises(ValueError, morgan_fingerprint.draw_bits, molecule, [1, 114, 227],
                          file_path=os.path.join(TEST_DIR, "data", 'test_morgan_fingerprint_draw_bit.svg'))

    def test_macckeys_fingerprint_draw_bit(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        maccs_keys_fingerprints = MACCSkeysFingerprint()
        maccs_keys_fingerprints.featurize(dataset)
        depiction = maccs_keys_fingerprints.draw_bit(molecule, 164, file_path=os.path.join(TEST_DIR, "data",
                                                                                           "test_macckeys_fingerprint_draw_bit.png"))

        self.assertTrue(isinstance(depiction, PngImageFile))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", "test_macckeys_fingerprint_draw_bit.png")))
        depiction.close()
        os.remove(os.path.join(TEST_DIR, "data", "test_macckeys_fingerprint_draw_bit.png"))

    def test_macckeys_fingerprint_draw_bit_to_fail(self):
        dataset = copy(self.mock_dataset_with_invalid)
        molecule = dataset.mols[0]
        maccs_keys_fingerprints = MACCSkeysFingerprint()
        maccs_keys_fingerprints.draw_bit(molecule, 66)
        self.assertRaises(ValueError, maccs_keys_fingerprints.draw_bit, molecule, 1)
        self.assertRaises(ValueError, maccs_keys_fingerprints.draw_bit, molecule, 66, "test.svg")
        self.assertRaises(ValueError, maccs_keys_fingerprints.draw_bit, molecule, 1, "test.png")
        self.assertRaises(ValueError, maccs_keys_fingerprints.draw_bit, molecule, -1, "test.png")
        molecule = dataset.mols[-1]
        self.assertRaises(ValueError, maccs_keys_fingerprints.draw_bit, molecule, 1, "test.png")

    def test_rdk_fingerprints_draw_bit(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        rdk_fingerprints = RDKFingerprint()
        rdk_fingerprints.draw_bit(molecule, 56,
                                  folder_path=os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints"))
        self.assertTrue(os.path.isdir(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints")))

        # Remove the folder and the contents inside
        shutil.rmtree(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints"))

    def test_rdk_fingerprints_draw_bits(self):
        dataset = copy(self.mock_dataset)
        molecule = dataset.mols[0]
        rdk_fingerprints = RDKFingerprint()
        images = rdk_fingerprints.draw_bits(molecule, [56, 61, 137], file_path=os.path.join(TEST_DIR, "data",
                                                                                            "maccs_keys_fingerprints.svg"))

        self.assertTrue(isinstance(images, SVG))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints.svg")))
        os.remove(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints.svg"))

    def test_rdk_fingerprints_bit_drawing_to_fail(self):
        dataset = copy(self.mock_dataset_with_invalid)
        molecule = dataset.mols[0]
        RDK_fingerprints = RDKFingerprint()
        RDK_fingerprints.draw_bit(molecule, 56)
        self.assertRaises(ValueError, RDK_fingerprints.draw_bit, molecule, 1)
        self.assertRaises(ValueError, RDK_fingerprints.draw_bit, molecule, 66, "test.svg")
        molecule = dataset.mols[-1]
        self.assertRaises(ValueError, RDK_fingerprints.draw_bit, molecule, 1, "test.png")

    def test_rdk_fingerprints_bits_drawing_to_fail(self):
        dataset = copy(self.mock_dataset_with_invalid)
        molecule = dataset.mols[0]
        print(dataset.smiles[0])
        RDK_fingerprints = RDKFingerprint()
        RDK_fingerprints.draw_bits(molecule, [56, 61, 137], file_path=os.path.join(TEST_DIR, "data",
                                                                                            "rdk_fingeprints.svg"))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", "rdk_fingeprints.svg")))
        os.remove(os.path.join(TEST_DIR, "data", "rdk_fingeprints.svg"))

        self.assertRaises(ValueError, RDK_fingerprints.draw_bits, molecule, [1, 2, 3])
        self.assertRaises(ValueError, RDK_fingerprints.draw_bits, molecule, [13, 20, 3])
        self.assertRaises(ValueError, RDK_fingerprints.draw_bits, molecule, "OFF")

        RDK_fingerprints.draw_bits(molecule, "ON", file_path=os.path.join(TEST_DIR, "data",
                                                                                   "maccs_keys_fingerprints.svg"))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints.svg")))
        os.remove(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints.svg"))

        RDK_fingerprints.draw_bits(molecule, 56, file_path=os.path.join(TEST_DIR, "data",
                                                                          "maccs_keys_fingerprints.svg"))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints.svg")))
        os.remove(os.path.join(TEST_DIR, "data", "maccs_keys_fingerprints.svg"))

        self.assertRaises(ValueError, RDK_fingerprints.draw_bits, molecule, 1, "test.png")

