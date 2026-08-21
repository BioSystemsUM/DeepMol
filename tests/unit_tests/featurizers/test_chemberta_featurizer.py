

from copy import copy
from unittest import TestCase
import numpy as np
from deepmol.compound_featurization.huggingface_featurizer import HuggingFaceFeaturizer
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
import unittest
from rdkit.Chem import MolFromSmiles


class TestHuggingFaceFeaturizer(FeaturizerTestCase, TestCase):
    
    def test_featurize(self):
        """Test featurization with valid molecules."""
        dataset_rows_number = len(self.mock_dataset.mols)
        HuggingFaceFeaturizer().featurize(self.mock_dataset, inplace=True)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

    def test_featurize_with_nan(self):
        """Test featurization with dataset containing invalid SMILES."""
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        HuggingFaceFeaturizer().featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_featurize_single_molecule(self):
        """Test featurization of a single molecule."""
        from rdkit.Chem import MolFromSmiles
        
        mol = MolFromSmiles("CCO")  # Ethanol
        featurizer = HuggingFaceFeaturizer()
        embedding = featurizer._featurize(mol)
        
        # Check embedding properties
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], len(featurizer.feature_names))
        self.assertFalse(np.isnan(embedding).all())

    def test_different_pooling_strategies(self):
        """Test different pooling strategies."""
        from rdkit.Chem import MolFromSmiles
        
        mol = MolFromSmiles("CCO")
        
        for pooling in ["mean", "cls"]:
            with self.subTest(pooling=pooling):
                featurizer = HuggingFaceFeaturizer(pooling=pooling)
                embedding = featurizer._featurize(mol)
                
                self.assertIsInstance(embedding, np.ndarray)
                self.assertEqual(embedding.shape[0], len(featurizer.feature_names))

    def test_features_names(self):
        """Test that feature names are properly set."""
        featurizer = HuggingFaceFeaturizer()
        self.assertIsNotNone(featurizer.feature_names)
        self.assertTrue(all(name.startswith('chemberta_') for name in featurizer.feature_names))

    def test_batch_featurize_valid_molecules(self):
        """Test batch featurization with valid molecules."""
        # Create a list of valid molecules
        from rdkit.Chem import MolFromSmiles
        molecules = [
            MolFromSmiles("CCO"),      # Ethanol
            MolFromSmiles("CCN"),      # Ethylamine
            MolFromSmiles("CCOC"),     # Dimethyl ether
            MolFromSmiles("CC(=O)O"),  # Acetic acid
            MolFromSmiles("c1ccccc1"), # Benzene
        ]
    
        featurizer = HuggingFaceFeaturizer(batch_size=2)  # Small batch size for testing
        embeddings = featurizer.batch_featurize(molecules)
    
        # Check shape
        self.assertEqual(embeddings.shape[0], len(molecules))
        self.assertEqual(embeddings.shape[1], len(featurizer.feature_names))
    
        # Check that all embeddings are valid (not NaN)
        self.assertFalse(np.isnan(embeddings).any())
    
        # Check that embeddings are different for different molecules
        self.assertFalse(np.array_equal(embeddings[0], embeddings[1]))

    def test_batch_featurize_with_invalid_molecules(self):
        """Test batch featurization with mixed valid and invalid molecules."""
        from rdkit.Chem import MolFromSmiles
        molecules = [
            MolFromSmiles("CCO"),      # Valid
            None,                       # Invalid
            MolFromSmiles("CCN"),      # Valid
            MolFromSmiles("INVALID"),  # Invalid (RDKit returns None)
            MolFromSmiles("CCOC"),     # Valid
        ]
        
        # Convert invalid SMILES to None
        molecules[3] = MolFromSmiles("INVALID")  # This should return None
        
        featurizer = HuggingFaceFeaturizer(batch_size=2)
        embeddings = featurizer.batch_featurize(molecules)
        
        # Check shape (should maintain original length)
        self.assertEqual(embeddings.shape[0], len(molecules))
        self.assertEqual(embeddings.shape[1], len(featurizer.feature_names))
        
        # Check that invalid molecules have NaN embeddings
        invalid_indices = [1, 3]  # Positions of invalid molecules
        for idx in invalid_indices:
            self.assertTrue(np.isnan(embeddings[idx]).all())
        
        # Check that valid molecules have non-NaN embeddings
        valid_indices = [0, 2, 4]
        for idx in valid_indices:
            self.assertFalse(np.isnan(embeddings[idx]).any())

    def test_batch_featurize_empty_list(self):
        """Test batch featurization with empty molecule list."""
        featurizer = HuggingFaceFeaturizer()
        embeddings = featurizer.batch_featurize([])
        
        # Should return empty array with correct dimensions
        self.assertEqual(embeddings.shape[0], 0)
        self.assertEqual(embeddings.shape[1], len(featurizer.feature_names))

    def test_batch_featurize_all_invalid(self):
        """Test batch featurization when all molecules are invalid."""
        molecules = [None, None, None]
        
        featurizer = HuggingFaceFeaturizer()
        embeddings = featurizer.batch_featurize(molecules)
        
        # Should return array of NaNs with correct shape
        self.assertEqual(embeddings.shape[0], len(molecules))
        self.assertEqual(embeddings.shape[1], len(featurizer.feature_names))
        self.assertTrue(np.isnan(embeddings).all())

    def test_batch_featurize_different_batch_sizes(self):
        """Test that different batch sizes produce same results."""
        from rdkit.Chem import MolFromSmiles
        
        molecules = [
            MolFromSmiles("CCO"), MolFromSmiles("CCN"), MolFromSmiles("CCOC"),
            MolFromSmiles("CC(=O)O"), MolFromSmiles("c1ccccc1"), MolFromSmiles("CC(C)C")
        ]
        
        # Test with different batch sizes
        for batch_size in [1, 2, 3, 6]:
            with self.subTest(batch_size=batch_size):
                featurizer = HuggingFaceFeaturizer(batch_size=batch_size)
                embeddings = featurizer.batch_featurize(molecules)
                
                # Should have correct shape and no NaN values
                self.assertEqual(embeddings.shape[0], len(molecules))
                self.assertEqual(embeddings.shape[1], len(featurizer.feature_names))
                self.assertFalse(np.isnan(embeddings).any())

    def test_batch_featurize_vs_individual(self):
        """Test that batch featurization produces same results as individual featurization."""
        from rdkit.Chem import MolFromSmiles
        
        molecules = [
            MolFromSmiles("CCO"), MolFromSmiles("CCN"), MolFromSmiles("CCOC"),
            MolFromSmiles("CC(=O)O"), MolFromSmiles("c1ccccc1")
        ]
        
        featurizer = HuggingFaceFeaturizer(batch_size=2)
        
        # Get batch embeddings
        batch_embeddings = featurizer.batch_featurize(molecules)
        
        # Get individual embeddings
        individual_embeddings = []
        for mol in molecules:
            embedding = featurizer._featurize(mol)
            individual_embeddings.append(embedding)
        individual_embeddings = np.array(individual_embeddings)
        
        # Should be very close (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(batch_embeddings, individual_embeddings, decimal=6)
