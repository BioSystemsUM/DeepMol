from unittest import TestCase

from deepmol.tokenizers.kmer_smiles_tokenizer import KmerSmilesTokenizer
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestKmersSmilesTokenizer(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        mock_dataset = self.mock_dataset.__copy__()
        tokenizer = KmerSmilesTokenizer(size=3, stride=1, n_jobs=-1)
        with self.assertRaises(ValueError):
            tokenizer.tokenize(mock_dataset)
        tokenizer = tokenizer.fit(mock_dataset)
        tokens = tokenizer.tokenize(mock_dataset)
        single_char_tokens = tokens[:2]
        for sct in single_char_tokens:
            # assert that each string in sct has length 3
            self.assertTrue(all(len(s) == 3 for s in sct))
        multi_char_tokens = tokens[3:]
        for mct in multi_char_tokens:
            # assert that at least one string in mct has length > 3
            self.assertTrue(any(len(s) > 3 for s in mct))

    def test_size_stride(self):
        mock_dataset = self.mock_dataset.__copy__()
        tokenizer = KmerSmilesTokenizer(size=4, stride=1, n_jobs=-1)
        with self.assertRaises(ValueError):
            tokenizer.tokenize(mock_dataset)
        tokenizer = tokenizer.fit(mock_dataset)
        tokens = tokenizer.tokenize(mock_dataset)
        single_char_tokens = tokens[:2]
        for sct in single_char_tokens:
            # assert that each string in sct has length 4
            self.assertTrue(all(len(s) == 4 for s in sct))
        multi_char_tokens = tokens[3:]
        for mct in multi_char_tokens:
            # assert that at least one string in mct has length > 4
            self.assertTrue(any(len(s) > 4 for s in mct))

        mock_dataset2 = self.mock_dataset.__copy__()
        tokenizer2 = KmerSmilesTokenizer(size=4, stride=2, n_jobs=-1)
        with self.assertRaises(ValueError):
            tokenizer2.tokenize(mock_dataset2)
        tokenizer2 = tokenizer2.fit(mock_dataset2)
        tokens2 = tokenizer2.tokenize(mock_dataset2)
        single_char_tokens2 = tokens2[:2]
        for sct in single_char_tokens2:
            # assert that each string in sct has length 4
            self.assertTrue(all(len(s) == 4 for s in sct))
        multi_char_tokens2 = tokens2[3:]
        for mct in multi_char_tokens2:
            # assert that at least one string in mct has length > 4
            self.assertTrue(any(len(s) > 4 for s in mct))

        self.assertGreater(sum(len(s) for s in tokens), sum(len(s) for s in tokens2))
        self.assertGreater(len(tokenizer.vocabulary), len(tokenizer2.vocabulary))
