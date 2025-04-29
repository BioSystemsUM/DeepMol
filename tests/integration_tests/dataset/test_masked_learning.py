


import os
import shutil
from unittest import TestCase, skip

from deepmol.models.transformer_models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
from tests.integration_tests.dataset.test_dataset import TestDataset

@skip
class TestMaskedLearningModels(TestDataset, TestCase):

    def test_fit_predict_evaluate(self):
        # ModernBERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # BERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # RoBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        model = DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)

        model.save("test")
        model = DeBERTa.load("test")
        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')

        if os.path.exists('test'):
            os.removedirs('test')

    def test_fine_tuning(self):
        self.dataset_for_masked_learning.mask = False
        model = DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", mode="classification",
                        devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        
        predictions = model.predict(self.dataset_for_masked_learning)

        self.assertEqual(predictions.shape[0], self.dataset_for_masked_learning.y.shape[0])
        self.assertEqual(predictions.shape[1], self.dataset_for_masked_learning.y.shape[1])

        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')

    def test_save_and_load(self):

        from torchmetrics.text import Perplexity

        for model_cls in [BERT, DeBERTa, ModernBERT]:

            self.dataset_for_masked_learning.mask = True
            
            model = model_cls(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size,
                            accelerator="cpu" , max_epochs=3, batch_size=56, metric = Perplexity())

            model.mode = "masked_learning"
            model.fit(self.dataset_for_masked_learning)
            print(model.evaluate(self.dataset_for_masked_learning))
            model.save("test")

            model = model_cls.load("test")
            model.mode = "classification"
            model.freeze_transformer = True
            model.metric = None

            if os.path.exists('test'):
                shutil.rmtree('test')

            self.dataset_for_masked_learning.mask = False
            model.trainer_kwargs["max_epochs"] = 3
            model.fit(self.dataset_for_masked_learning)

            predictions = model.predict(self.dataset_for_masked_learning)

            self.assertEqual(predictions.shape[0], self.dataset_for_masked_learning.y.shape[0])
            self.assertEqual(predictions.shape[1], self.dataset_for_masked_learning.y.shape[1])

            model.save("test2")
            model = model_cls.load("test2")
            
            if os.path.exists('vocab.txt'):
                os.remove('vocab.txt')

            if os.path.exists('test2'):
                shutil.rmtree('test2')

            predictions_2 = model.predict(self.dataset_for_masked_learning)

            self.assertEqual(predictions_2.shape[0], self.dataset_for_masked_learning.y.shape[0])
            self.assertEqual(predictions_2.shape[1], self.dataset_for_masked_learning.y.shape[1])

            import numpy as np
            np.testing.assert_array_equal(predictions, predictions_2)
        

        
