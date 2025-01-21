


import os
from unittest import TestCase, skip

from deepmol.models.transformer_models import BERT, ModernBERT, RoBERTa, DeBERTa
from tests.integration_tests.dataset.test_dataset import TestDataset

@skip
class TestMaskedLearningModels(TestDataset, TestCase):

    def test_fit_predict_evaluate(self):
        # ModernBERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # BERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # RoBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')