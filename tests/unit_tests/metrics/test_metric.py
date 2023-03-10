from unittest import TestCase

import numpy as np

from deepmol.metrics import Metric


class TestMetric(TestCase):

    def test_metric(self):
        def metric_test(y_true_, y_pred_):
            if len(y_true_.shape) == 1:
                score = 0
                for i in range(len(y_true_)):
                    if y_true_[i] ** 2 == y_pred_[i] ** 2:
                        score += 1
                return score / len(y_true_)
            else:
                score = 0
                n = 0
                for i in range(len(y_true_)):
                    for j in range(len(y_true_[i])):
                        n += 1
                        if y_true_[i][j] ** 2 == y_pred_[i][j] ** 2:
                            score += 1
                return score / n

        def task_averager_test(task_metrics):
            return (sum(task_metrics) + 1) / len(task_metrics)

        metric = Metric(metric_test)

        y_true = np.array([0, 1, 0, 1, 0, 1, 0, -1, 0, 1])
        y_pred = np.array([0, -1, 0, 1, 0, 1, 0, 1, 0, -1])

        metric_value = metric.compute_metric(y_true, y_pred, n_tasks=1)[0]
        self.assertEqual(metric_value, 1.0)

        metric_2 = Metric(metric_test,
                          name='test_metric',
                          task_averager=task_averager_test)

        y_true_2 = np.array([[0, 1, -1],
                             [1, 0, 1],
                             [0, 1, 0]])
        y_pred_2 = np.array([[1, -1, 1],
                             [1, 1, 1],
                             [1, 1, 0]])

        metric_value_2 = metric_2.compute_metric(y_true_2, y_pred_2, n_tasks=3, per_task_metrics=True)
        self.assertEqual(metric_value_2[0], 2/3)
        self.assertEqual(metric_value_2[1], [1/3, 2/3, 1])

        def single_task_metric(y_true_, y_pred_):
            score = 0
            for i in range(len(y_true_)):
                if y_true_[i] ** 2 == y_pred_[i] ** 2:
                    score += 1
            return score / len(y_true_)

        metric_3 = Metric(single_task_metric, task_averager=task_averager_test)
        metric_value_3 = metric_3.compute_metric(y_true_2, y_pred_2, n_tasks=3, per_task_metrics=True)
        self.assertEqual(metric_value_3[0], 1)
        self.assertEqual(metric_value_3[1], [1/3, 2/3, 1])
