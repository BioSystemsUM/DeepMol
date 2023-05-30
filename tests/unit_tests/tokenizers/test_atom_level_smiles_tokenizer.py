from unittest import TestCase

from deepmol.tokenizers import AtomLevelSmilesTokenizer
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestAtomLevelSmilesTokenizer(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        mock_dataset = self.mock_dataset.__copy__()
        tokenizer = AtomLevelSmilesTokenizer(n_jobs=-1)
        with self.assertRaises(ValueError):
            tokenizer.tokenize(mock_dataset)
        tokenizer = tokenizer.fit(mock_dataset)
        tokens = tokenizer.tokenize(mock_dataset)
        vocabulary = tokenizer.vocabulary
        single_char_tokens = tokens[:2]
        for sct in single_char_tokens:
            # assert that each string in sct has length 1
            self.assertTrue(all(len(s) == 1 for s in sct))
        multi_char_tokens = tokens[3:]
        for mct in multi_char_tokens:
            # assert that at least one string in mct has length > 1
            self.assertTrue(any(len(s) > 1 for s in mct))

        br_smiles = tokens[-2:]
        self.assertTrue('Br' in br_smiles[0] or 'Br' in br_smiles[1])

        self.assertTrue('[C@@H]' in tokens[3] and '[C@H]' in tokens[4])

        tokenizer.regex = "(Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        tokenizer = tokenizer.fit(mock_dataset)
        tokens2 = tokenizer.tokenize(mock_dataset)
        vocabulary2 = tokenizer.vocabulary
        self.assertFalse('[C@@H]' in tokens2[3] and '[C@H]' in tokens2[4])

        self.assertGreater(len(vocabulary), len(vocabulary2))
        self.assertTrue('[C@@H]' in vocabulary and '[C@H]' not in vocabulary2)
