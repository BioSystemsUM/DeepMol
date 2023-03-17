from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report, \
    mean_squared_error, mean_absolute_error, f1_score, r2_score, explained_variance_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from unit_tests.models.test_models import ModelsTestCase


class TestSklearnModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)
        model.fit(self.binary_dataset)

        test_preds = model.predict(self.binary_dataset_test)
        self.assertEqual(len(test_preds), len(self.binary_dataset_test))
        # evaluate the model
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = model.evaluate(self.binary_dataset_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})
        self.assertTrue('roc_auc_score' in evaluate[0].keys())
        self.assertTrue('precision_score' in evaluate[0].keys())
        self.assertTrue('accuracy_score' in evaluate[0].keys())
        self.assertTrue('confusion_matrix' in evaluate[0].keys())
        self.assertTrue('classification_report' in evaluate[0].keys())

    def test_aucs(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)
        model.fit(self.binary_dataset)
        metrics = [Metric(roc_auc_score)]

        # evaluate the model
        train_score = model.evaluate(self.binary_dataset, metrics)

        y_pred = model.model.predict_proba(self.binary_dataset.X)
        y_pred = y_pred[:, 1]
        roc_auc_score_value = roc_auc_score(self.binary_dataset.y, y_pred)
        self.assertEqual(roc_auc_score_value, train_score[0]["roc_auc_score"])

    def test_regression(self):
        rf = RandomForestRegressor()
        model = SklearnModel(model=rf)
        model.fit(self.regression_dataset)
        metrics = [Metric(mean_squared_error), Metric(mean_absolute_error)]

        # evaluate the model
        test_score = model.evaluate(self.regression_dataset_test, metrics)

        y_pred = model.model.predict(self.regression_dataset_test.X)
        mean_squared_error_value = mean_squared_error(self.regression_dataset_test.y, y_pred)
        self.assertEqual(mean_squared_error_value, test_score[0]["mean_squared_error"])
        mean_absolute_error_value = mean_absolute_error(self.regression_dataset_test.y, y_pred)
        self.assertEqual(mean_absolute_error_value, test_score[0]["mean_absolute_error"])

    def test_multiclass_classification(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)
        model.fit(self.multiclass_dataset)
        metrics = [Metric(accuracy_score)]

        # evaluate the model
        test_score = model.evaluate(self.multiclass_dataset_test, metrics)

        y_pred = model.model.predict(self.multiclass_dataset_test.X)
        accuracy_score_value = accuracy_score(self.multiclass_dataset_test.y, y_pred)
        self.assertEqual(accuracy_score_value, test_score[0]["accuracy_score"])

    def test_multitask_classification(self):
        forest = RandomForestClassifier(random_state=1)
        multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
        model = SklearnModel(model=multi_target_forest)
        model.fit(self.multitask_dataset)
        metrics = [Metric(precision_score, average='samples'), Metric(f1_score, average='samples')]

        # evaluate the model
        test_score = model.evaluate(self.multitask_dataset_test, metrics, per_task_metrics=True)

        y_pred = model.model.predict(self.multitask_dataset_test.X)
        precision_score_value = precision_score(self.multitask_dataset_test.y, y_pred, average='samples')
        f1_score_value = f1_score(self.multitask_dataset_test.y, y_pred, average='samples')
        self.assertEqual(precision_score_value, test_score[0]["precision_score"])
        self.assertEqual(f1_score_value, test_score[0]["f1_score"])

    def test_multitask_regression(self):
        model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        model = SklearnModel(model=model)
        model.fit(self.multitask_regression_dataset)
        metrics = [Metric(mean_squared_error), Metric(mean_absolute_error), Metric(r2_score)]

        # evaluate the model
        test_score = model.evaluate(self.multitask_regression_dataset_test, metrics)

        y_pred = model.model.predict(self.multitask_regression_dataset_test.X)
        mean_squared_error_value = mean_squared_error(self.multitask_regression_dataset_test.y, y_pred)
        mean_absolute_error_value = mean_absolute_error(self.multitask_regression_dataset_test.y, y_pred)
        r2_score_value = r2_score(self.multitask_regression_dataset_test.y, y_pred)
        explained_variance_score_value = explained_variance_score(self.multitask_regression_dataset_test.y, y_pred)
        self.assertEqual(mean_squared_error_value, test_score[0]["mean_squared_error"])
        self.assertEqual(mean_absolute_error_value, test_score[0]["mean_absolute_error"])
        self.assertEqual(r2_score_value, test_score[0]["r2_score"])
