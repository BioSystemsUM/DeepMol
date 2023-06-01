from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization._deepchem_models_objectives import gat_model_steps, gcn_model_steps, \
    pagtn_model_steps, attentive_fp_model_steps, mpnn_model_steps, megnet_model_steps, cnn_model_steps
from deepmol.pipeline_optimization._featurizer_objectives import _get_featurizer
from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer


# TODO: add more models
_BASE_MODELS = []
_RNN_MODELS = []
_CNN_MODELS = []


# TODO: How to deal with incompatible steps? (e.g. some deepchem featurizers only work with some deepchem models)


def _get_preset(preset: str) -> callable:
    return test_objective_classification


def test_objective_classification(trial,
                                  train_dataset: Dataset,
                                  test_dataset: Dataset,
                                  metric: Metric) -> callable:
    # TODO: "mpnn_model" is not working (DeepChem-> AttributeError: 'GraphData' object has no attribute 'get_num_atoms')
    # TODO: "megnet_model" is not working (error with torch_geometric (extra_requirement))
    # TODO: "cnn_model" is not working (raise PicklingError(
    #  pickle.PicklingError: Can't pickle <class '_thread.lock'>: it's not found as _thread.lock))
    # steps = trial.suggest_categorical("steps", ["gat_model", "gcn_model", "attentive_fp_model", "pagtn_model",
    #                                             "cnn_model"])
    steps = trial.suggest_categorical("steps", ["cnn_model"])
    if steps == "gat_model":
        steps = gat_model_steps(trial=trial, n_tasks=1, mode='classification')
    elif steps == "gcn_model":
        steps = gcn_model_steps(trial=trial, n_tasks=1, mode='classification')
    elif steps == "attentive_fp_model":
        steps = attentive_fp_model_steps(trial=trial, n_tasks=1, mode='classification')
    elif steps == "pagtn_model":
        steps = pagtn_model_steps(trial=trial, n_tasks=1, mode='classification')
    # elif steps == "mpnn_model":
    #     steps = mpnn_model_steps(trial=trial, n_tasks=1, mode='classification')
    # elif steps == "megnet_model":
    #     steps = megnet_model_steps(trial=trial, n_tasks=1, mode='classification')
    # elif steps == "cnn_model":
    #     n_features = 1024
    #     dims = 1
    #     featurizer = MorganFingerprint(size=1024)
    #     steps = cnn_model_steps(trial=trial, n_features=n_features, dims=dims, n_tasks=1, mode='classification')
    #     steps = (('featurizer', featurizer), steps[0])
    print(trial.params)
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, [metric])[0][metric.name]
    return score


def heavy_objective(trial, train_dataset, test_dataset, metric) -> callable:
    # model
    model = trial.suggest_categorical("model", _BASE_MODELS)
    # standardizer
    standardizer = _get_standardizer(trial)
    # featurizer
    featurizer = _get_featurizer(trial, '1D')
    # scaler
    scaler = trial.suggest_categorical("scaler", _SCALERS)
    # feature_selector
    feature_selector = trial.suggest_categorical("feature_selector", _FEATURE_SELECTORS)
    steps = [('standardizer', standardizer), ('featurizer', featurizer), ('scaler', scaler),
             ('feature_selector', feature_selector), ('model', model)]
    pipeline = Pipeline(steps=steps)
    pipeline.fit_transform(train_dataset)
    score = pipeline.evaluate(test_dataset, metric)[0][metric.name]
    return score
