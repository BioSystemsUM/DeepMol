import os
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from deepmol.datasets import Dataset
from deepmol.evaluator import Evaluator
from deepmol.metrics import Metric
from deepmol.models import Model


class TestEvaluator(TestCase):

    def tearDown(self) -> None:
        if os.path.exists('test.txt'):
            os.remove('test.txt')
        if os.path.exists('test2.txt'):
            os.remove('test2.txt')

    def test_evaluator(self):
        dataset = MagicMock(spec=Dataset,
                            ids=np.array([1, 2, 3]),
                            y=np.array([1, 2, 3]),
                            n_tasks=1,
                            label_names=np.array(['label']))

        model = MagicMock(spec=Model)
        model.predict.return_value = [1, 2, 3]

        metric1 = MagicMock(spec=Metric)
        metric1.name = 'metric1'
        metric1.compute_metric.return_value = (0.5, None)

        metric2 = MagicMock(spec=Metric)
        metric2.name = 'metric2'
        metric2.compute_metric.return_value = (0.25, None)

        evaluator = Evaluator(model, dataset)

        multitask_scores, all_task_scores = evaluator.compute_model_performance(metrics=[metric1, metric2],
                                                                                per_task_metrics=True)

        self.assertEqual(multitask_scores, {'metric1': 0.5, 'metric2': 0.25})
        self.assertEqual(all_task_scores, {'metric1': None, 'metric2': None})

        evaluator.output_statistics(multitask_scores, 'test.txt')
        evaluator.output_predictions(np.array([1, 2, 3]), 'test2.txt')

        self.assertTrue(os.path.exists('test.txt'))
        self.assertTrue(os.path.exists('test2.txt'))
