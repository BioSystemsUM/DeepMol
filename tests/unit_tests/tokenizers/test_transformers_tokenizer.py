from copy import copy
import os
from unittest import TestCase

from deepmol.tokenizers.kmer_smiles_tokenizer import KmerSmilesTokenizer
from deepmol.tokenizers.transformers_tokenizer import SmilesTokenizer
from tests import TEST_DIR
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestTokenizersTokenizer(FeaturizerTestCase, TestCase):

    def test_export_vocabulary(self):
        vocab_path = os.path.join(TEST_DIR,"data", 'vocab.txt')
        SmilesTokenizer.export_vocab(self.mock_dataset, vocab_path)

    def test_featurize(self):
        mock_dataset = copy(self.mock_dataset)
        vocab_path = os.path.join(TEST_DIR, "data", 'vocab.txt')
        print(SmilesTokenizer(vocab_path)._tokenize(mock_dataset.smiles[0]))

