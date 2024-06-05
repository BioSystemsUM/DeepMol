import os
import shutil
from unittest import TestCase

import numpy as np
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset
from deepmol.loggers import Logger
from deepmol.pipeline import Pipeline
from tests.unit_tests._mock_utils import SmilesDatasetMagicMock, MockTransformerMagicMock, MockPredictorMagicMock, \
    MockMetricMagicMock


class TestPipeline(TestCase):

    def setUp(self) -> None:
        # create a dataset with 14 samples and 5 features
        x = np.random.randint(0, 10, size=(14, 5))
        # y with 10 samples of class 0 and 4 samples of class 1
        y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        # ids 10 characters from the alphabet
        ids = np.array([''.join(np.random.choice(list('abcdefghij'), 14)) for _ in range(14)])
        # smiles with one more C than the last one
        smiles = np.array(['C' * (i + 1) for i in range(14)])
        # mols
        mols = np.array([MolFromSmiles(s) for s in smiles])
        self.dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                              X=x,
                                              smiles=smiles,
                                              mols=mols,
                                              y=y,
                                              ids=ids)
        self.dataset.__len__.return_value = 14

        self.mock_transformer1 = MockTransformerMagicMock(value=1)
        self.mock_transformer2 = MockTransformerMagicMock(value=2)
        self.mock_predictor = MockPredictorMagicMock(seed=123)

        self.metric1 = MockMetricMagicMock(name='metric1')
        self.metric2 = MockMetricMagicMock(name='metric2')

        self.pipeline_path = 'tests/unit_tests/test_pipeline/'

    def tearDown(self) -> None:
        # Close logger file handlers to release the file
        singleton_instance = Logger()
        singleton_instance.close_handlers()

        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')
        if os.path.exists(self.pipeline_path):
            shutil.rmtree(self.pipeline_path)

    def test_pipeline(self):
        steps = [('mock_transformer1', self.mock_transformer1),
                 ('mock_transformer2', self.mock_transformer2),
                 ('mock_predictor', self.mock_predictor)]
        pipeline = Pipeline(steps=steps, path=self.pipeline_path)
        self.assertTrue(pipeline.is_prediction_pipeline())
        self.assertFalse(pipeline.is_fitted())

        pipeline.fit(self.dataset)
        self.assertTrue(pipeline.is_fitted())

        predictions = pipeline.predict(self.dataset)
        self.assertEqual(len(predictions), self.dataset.__len__())
        e1, e2 = pipeline.evaluate(self.dataset, [self.metric1, self.metric2])
        self.assertTrue('metric1' in e1.keys())
        self.assertTrue('metric2' in e1.keys())
        self.assertEqual(e2, {})

        predict_proba = pipeline.predict_proba(self.dataset)
        self.assertEqual(len(predict_proba), self.dataset.__len__())
        self.assertEqual(predict_proba.shape[1], 2)
