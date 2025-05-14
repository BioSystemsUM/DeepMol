from copy import copy
import os
from deepmol.compound_featurization.llms import LLM
from deepmol.tokenizers.transformers_tokenizer import SmilesTokenizer
from tests import TEST_DIR
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from rdkit.Chem import MolFromSmiles
from transformers import ModernBertModel, ModernBertConfig

from deepmol.compound_featurization.biosynfoni import BiosynfoniKeys


class TestLLMs(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        model_path = os.path.join(TEST_DIR, "data/llm_model")
        transformer = LLM(model_path=model_path, model=ModernBertModel, config_class=ModernBertConfig,
                          tokenizer=SmilesTokenizer(vocab_file=os.path.join(model_path, "vocab.txt")))
        transformer.featurize(self.mock_dataset, inplace=True)
        self.assertEqual(7, self.mock_dataset._X.shape[0])
        self.assertEqual(256, self.mock_dataset._X.shape[1])

    def test_featurize_with_nan(self):
        model_path = os.path.join(TEST_DIR, "data/llm_model")
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols)  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        LLM(model_path=model_path, model=ModernBertModel, config_class=ModernBertConfig,
                          tokenizer=SmilesTokenizer(vocab_file=os.path.join(model_path, "vocab.txt"))).featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_fingerprint_with_no_ones(self):
        model_path = os.path.join(TEST_DIR, "data/llm_model")
        transformer = LLM(model_path=model_path, model=ModernBertModel, config_class=ModernBertConfig,
                          tokenizer=SmilesTokenizer(vocab_file=os.path.join(model_path, "vocab.txt")))
        bitstring = transformer._featurize("COC1=CC(=CC(=C1OC)OC)CCN")
        self.assertEqual(bitstring.shape[0], 256)