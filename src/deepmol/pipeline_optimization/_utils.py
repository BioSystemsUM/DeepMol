from copy import deepcopy, copy
from typing import Literal

try:
    from deepchem.models import TextCNNModel
    from deepmol.pipeline_optimization._keras_model_objectives import _get_keras_model
    from deepmol.pipeline_optimization._deepchem_models_objectives import *
    from deepmol.compound_featurization import CoulombFeat
except ImportError:
    pass

from deepmol.base import DatasetTransformer, PassThroughTransformer
from deepmol.datasets import Dataset
from deepmol.datasets._utils import _get_n_classes
from deepmol.encoders.label_one_hot_encoder import LabelOneHotEncoder
from deepmol.pipeline_optimization._feature_selector_objectives import _get_feature_selector
from deepmol.pipeline_optimization._featurizer_objectives import _get_featurizer
from deepmol.pipeline_optimization._scaler_objectives import _get_scaler
from deepmol.pipeline_optimization._sklearn_model_objectives import _get_sk_model
from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer
from deepmol.scalers.sklearn_scalers import MinMaxScaler


def _get_preset(preset: Literal['deepchem', 'sklearn', 'keras', 'all']) -> callable:
    """
    Returns the function that returns the list of steps for the given preset.

    Parameters
    ----------
    preset : Literal['deepchem', 'sklearn', 'keras', 'all']
        The preset to use.

    Returns
    -------
    callable
        The function that returns the list of steps for the given preset.
    """
    if preset == 'deepchem':
        return preset_deepchem_models
    elif preset == 'sklearn':
        return preset_sklearn_models
    elif preset == 'keras':
        return preset_keras_models
    elif preset == 'all':
        return preset_all_models


def _check_class_name_and_attribute_scaler_feature_selection(trial, featurizer):
    """
    Check if the class name and attribute of the scaler and feature selection are correct.
    """
    if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
            featurizer.__class__.__name__ == 'All3DDescriptors':
        transformer = ("scaler", _get_scaler(trial))
    else:
        transformer = ("scaler", PassThroughTransformer())

    return transformer


def _cnn_model_steps(trial, n_tasks, n_classes):
    featurizer = _get_featurizer(trial, '1D')

    transformer = _check_class_name_and_attribute_scaler_feature_selection(trial, featurizer)
    dims = 1
    n_features = len(featurizer.feature_names)
    cnn_kwargs = {'n_tasks': n_tasks, 'mode': 'regression', 'n_features': n_features, 'dims': dims,
                  'n_classes': n_classes}
    model_step = cnn_model_steps(trial=trial, cnn_kwargs=cnn_kwargs)
    featurizer = ('featurizer', featurizer)
    model = model_step[0]
    return [featurizer, transformer, model]


def _multitask_classifier_model_steps(trial, n_tasks, n_classes):
    featurizer = _get_featurizer(trial, '1D')
    transformer = _check_class_name_and_attribute_scaler_feature_selection(trial, featurizer)

    n_features = len(featurizer.feature_names)
    multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes}
    model_step = multitask_classifier_model_steps(trial=trial,
                                                  multitask_classifier_kwargs=multitask_classifier_kwargs)
    featurizer = ('featurizer', featurizer)
    model = model_step[0]
    return [featurizer, transformer, model]


def _progressive_multitask_regressor_model_steps(trial, n_tasks):
    featurizer = _get_featurizer(trial, '1D')

    transformer = _check_class_name_and_attribute_scaler_feature_selection(trial, featurizer)

    n_features = len(featurizer.feature_names)
    progressive_multitask_regressor_kwargs = {'n_tasks': n_tasks, 'n_features': n_features}
    model_step = progressive_multitask_regressor_model_steps(trial=trial,
                                                             progressive_multitask_regressor_kwargs=progressive_multitask_regressor_kwargs)
    featurizer = ('featurizer', featurizer)
    return [featurizer, transformer, model_step[0]]


def _robust_multitask_regressor_model_steps(trial, n_tasks):
    featurizer = _get_featurizer(trial, '1D')

    transformer = _check_class_name_and_attribute_scaler_feature_selection(trial, featurizer)

    n_features = len(featurizer.feature_names)
    robust_multitask_regressor_kwargs = {'n_tasks': n_tasks, 'n_features': n_features}
    model_step = robust_multitask_regressor_model_steps(trial=trial,
                                                        robust_multitask_regressor_kwargs=robust_multitask_regressor_kwargs)
    featurizer = ('featurizer', featurizer)
    return [featurizer, transformer, model_step[0]]


def _multitask_regressor_model_steps(trial, n_tasks):
    featurizer = _get_featurizer(trial, '1D')

    transformer = _check_class_name_and_attribute_scaler_feature_selection(trial, featurizer)

    n_features = len(featurizer.feature_names)
    multitask_regressor_kwargs = {'n_tasks': n_tasks, 'n_features': n_features}
    model_step = multitask_regressor_model_steps(trial=trial,
                                                 multitask_regressor_kwargs=multitask_regressor_kwargs)
    featurizer = ('featurizer', featurizer)
    model = model_step[0]
    return [featurizer, transformer, model]


def preset_deepchem_models(trial, data: Dataset) -> list:
    """
    Returns the list of steps for the deepchem preset.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The trial.
    data : deepmol.datasets.Dataset
        The dataset.

    Returns
    -------
    list
        The list of steps for the deepchem preset.
    """
    # TODO: "mpnn_model" is not working (DeepChem-> AttributeError: 'GraphData' object has no attribute 'get_num_atoms')
    # TODO: "megnet_model" is not working (error with torch_geometric (extra_requirement))
    # TODO: "cnn_model" is not working (raise PicklingError(
    #  pickle.PicklingError: Can't pickle <class '_thread.lock'>: it's not found as _thread.lock))
    # TODO: "multitask_irv_classifier_model" not working (needs 1D featurizer + irv transformer but it is not working)
    # TODO: "progressive_multitask_classifier_model" not working (ValueError: Index out of range using input dim 1;
    #  input has only 1 dims for '{{node strided_slice_1}} ...) (SAME WITH progressive_multitask_classifier_model)
    # TODO: "sc_score_model" not working (Input 0 of layer "dense" is incompatible with the layer: expected axis -1 of
    #  input shape to have value 2048, but received input with shape (100, 1))
    # TODO: "robust_multitask_classifier_model" works well except when loading the model back in (probably related with
    #  the custom_objects) (SAME WITH robust_multitask_regressor_model)
    n_tasks = data.n_tasks
    mode = data.mode
    n_classes = len(set(data.y)) if mode == 'classification' else 1
    if isinstance(mode, list):
        if mode[0] == "classification":
            n_classes = len(set(data.y[0]))
        else:
            n_classes = 1
    final_steps = [('standardizer', _get_standardizer(trial))]
    models = ["gat_model", "gcn_model", "attentive_fp_model", "pagtn_model", "chem_ception_model", "dag_model",
              "graph_conv_model", "smiles_to_vec_model", "text_cnn_model", "weave_model", "dmpnn_model"]
    batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
    epochs = trial.suggest_int("epochs_deepchem", 10, 200)
    deepchem_kwargs = {"epochs": epochs}
    if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
        if n_classes > 2:
            models.remove("text_cnn_model")
        if (len(set(mode)) == 1 and mode[0] == 'classification'):
            models.extend(["multitask_classifier_model", "robust_multitask_classifier_model"])
        mode = 'classification'
    elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
        models.extend(["dtnn_model", "mat_model"])  # ,
        # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
        mode = 'regression'
        if (len(set(mode)) == 1 and mode[0] == 'regression'):
            models.extend(["multitask_regressor_model"])
    else:
        raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

    model_steps = trial.suggest_categorical("model_steps", models)

    if model_steps == "gat_model":
        gat_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_gat = gat_model_steps(trial=trial, gat_kwargs=gat_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_gat)

    elif model_steps == "gcn_model":
        gcn_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_gcn = gcn_model_steps(trial=trial, gcn_kwargs=gcn_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_gcn)

    elif model_steps == "attentive_fp_model":
        attentive_fp_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_attentive = attentive_fp_model_steps(trial=trial, attentive_fp_kwargs=attentive_fp_kwargs,
                                                   deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_attentive)

    elif model_steps == "pagtn_model":
        patgn_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_patgn = pagtn_model_steps(trial=trial, pagtn_kwargs=patgn_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_patgn)

    elif model_steps == "mpnn_model":
        mpnn_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_mpnn = mpnn_model_steps(trial=trial, mpnn_kwargs=mpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_mpnn)

    elif model_steps == "megnet_model":
        megnet_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_megnet = megnet_model_steps(trial=trial, megnet_kwargs=megnet_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_megnet)

    elif model_steps == "cnn_model":
        featurizer_scaler_model = _cnn_model_steps(trial, n_tasks, n_classes)
        final_steps.extend(featurizer_scaler_model)

    elif model_steps == "multitask_classifier_model":
        featurizer_scaler_model = _multitask_classifier_model_steps(trial, n_tasks, n_classes)
        final_steps.extend(featurizer_scaler_model)

    elif model_steps == "multitask_irv_classifier_model":
        featurizer = _get_featurizer(trial, '1D')
        multitask_irv_classifier_kwargs = {'n_tasks': n_tasks, 'n_classes': n_classes}
        model_step = multitask_irv_classifier_model_steps(trial=trial,
                                                          multitask_irv_classifier_kwargs=multitask_irv_classifier_kwargs)
        featurizer = ('featurizer', featurizer)
        final_steps.extend([featurizer, model_step[0]])

    elif model_steps == "progressive_multitask_classifier_model":
        featurizer = _get_featurizer(trial, '1D')
        n_features = len(featurizer.feature_names)
        progressive_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                                   'batch_size': batch_size}
        model_step = progressive_multitask_classifier_model_steps(trial=trial,
                                                                  progressive_multitask_classifier_kwargs=progressive_multitask_classifier_kwargs,
                                                                  deepchem_kwargs=deepchem_kwargs)
        featurizer = ('featurizer', featurizer)
        final_steps.extend([featurizer, model_step[0]])
    elif model_steps == "robust_multitask_classifier_model":
        featurizer = _get_featurizer(trial, '1D')
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}
        model_step = robust_multitask_classifier_model_steps(trial=trial,
                                                             robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                                             deepchem_kwargs=deepchem_kwargs)
        featurizer = ('featurizer', featurizer)
        final_steps.extend([featurizer, model_step[0]])

    elif model_steps == "sc_score_model":
        featurizer = _get_featurizer(trial, '1D')
        n_features = len(featurizer.feature_names)
        sc_score_kwargs = {'n_features': n_features, 'n_classes': n_classes}
        model_step = sc_score_model_steps(trial=trial, sc_score_kwargs=sc_score_kwargs)
        featurizer = ('featurizer', featurizer)
        final_steps.extend([featurizer, model_step[0]])

    elif model_steps == "chem_ception_model":
        chem_ception_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_chem_ception = chem_ception_model_steps(trial=trial, chem_ception_kwargs=chem_ception_kwargs,
                                                      deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_chem_ception)

    elif model_steps == "dag_model":
        dag_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_dag = dag_model_steps(trial=trial, dag_kwargs=dag_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_dag)

    elif model_steps == "graph_conv_model":
        graph_conv_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_graph_conv = graph_conv_model_steps(trial=trial, graph_conv_kwargs=graph_conv_kwargs,
                                                  deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_graph_conv)

    elif model_steps == "smiles_to_vec_model":
        smiles_to_vec_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        dataset_copy = deepcopy(data)
        ssf = SmilesSeqFeat()
        ssf.fit_transform(dataset_copy)
        chat_to_idx = ssf.char_to_idx
        smiles_to_vec_kwargs['char_to_idx'] = chat_to_idx
        steps_smiles_to_vec = smiles_to_vec_model_steps(trial=trial, smiles_to_vec_kwargs=smiles_to_vec_kwargs,
                                                        deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_smiles_to_vec)

    elif model_steps == "text_cnn_model":
        text_cnn_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'batch_size': batch_size}
        max_length = max([len(smile) for smile in data.smiles])
        padded_train_smiles = prepare_dataset_for_textcnn(data, max_length).ids
        fake_dataset = copy(data)
        fake_dataset._ids = padded_train_smiles
        char_dict, seq_length = TextCNNModel.build_char_dict(fake_dataset)
        text_cnn_kwargs['char_dict'] = char_dict
        text_cnn_kwargs['seq_length'] = seq_length
        padder = DatasetTransformer(prepare_dataset_for_textcnn, max_length=max_length)
        final_steps.extend([('padder', padder), text_cnn_model_steps(trial=trial, text_cnn_kwargs=text_cnn_kwargs,
                                                                     deepchem_kwargs=deepchem_kwargs)[0]])

    elif model_steps == "weave_model":
        weave_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_weave = weave_model_steps(trial=trial, weave_kwargs=weave_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_weave)

    elif model_steps == "dmpnn_model":
        dmpnn_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        steps_dmpnn = dmpnn_model_steps(trial=trial, dmpnn_kwargs=dmpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(steps_dmpnn)

    elif model_steps == "progressive_multitask_regressor_model":
        featurizer_scaler_model_step = _progressive_multitask_regressor_model_steps(trial=trial, n_tasks=n_tasks)
        final_steps.extend(featurizer_scaler_model_step)

    elif model_steps == "robust_multitask_regressor_model":
        featurizer_scaler_model_step = _robust_multitask_regressor_model_steps(trial=trial, n_tasks=n_tasks)
        final_steps.extend(featurizer_scaler_model_step)

    elif model_steps == "dtnn_model":
        max_atoms = max([mol.GetNumAtoms() for mol in data.mols])
        featurizer = CoulombFeat(max_atoms=max_atoms)
        dtnn_kwargs = {'n_tasks': n_tasks, 'batch_size': batch_size}
        model_step = dtnn_model_steps(trial=trial, dtnn_kwargs=dtnn_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend([('featurizer', featurizer), model_step[0]])

    elif model_steps == "mat_model":
        mat_kwargs = {"batch_size": batch_size}
        mat_steps = mat_model_steps(trial=trial, mat_kwargs=mat_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend(mat_steps)

    elif model_steps == "multitask_regressor_model":
        featurizer_scaler_model_step = _multitask_regressor_model_steps(trial=trial, n_tasks=n_tasks)
        final_steps.extend(featurizer_scaler_model_step)
    else:
        raise ValueError("Unknown model: %s" % model_steps)
    return final_steps


def preset_sklearn_models(trial, data: Dataset) -> list:
    """
    Preset sklearn models for hyperparameter optimization.

    Parameters
    ----------
    trial: optuna.trial.Trial
        Trial object that stores the current progress of hyperparameter optimization.
    data: Dataset
        Dataset object.

    Returns
    -------
    final_steps: list
        List of tuples, where each tuple is a step in the sklearn pipeline.
    """
    mode = data.mode
    multitask = True if data.n_tasks > 1 else False
    featurizer = _get_featurizer(trial, '1D')
    if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
            featurizer.__class__.__name__ == 'All3DDescriptors':
        scaler = _get_scaler(trial)
    else:
        scaler = PassThroughTransformer()

    if featurizer.__class__.__name__ == 'nc_mfp':
        feature_selector = _get_feature_selector(trial, task_type=mode, multitask=multitask)
        while feature_selector.__class__.__name__ == 'PassThroughTransformer':
            feature_selector = _get_feature_selector(trial, task_type=mode, multitask=multitask)
    else:
        feature_selector = _get_feature_selector(trial, task_type=mode, multitask=multitask)

    if mode == 'classification':
        sk_mode = 'classification_binary' if set(data.y) == {0, 1} else 'classification_multiclass'
    else:
        sk_mode = mode
    sk_model = _get_sk_model(trial, task_type=sk_mode)
    if sk_model.model.__class__.__name__ == 'BernoulliNB' or sk_model.model.__class__.__name__ == 'MultinomialNB' or \
            sk_model.model.__class__.__name__ == 'ComplementNB':
        if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
                featurizer.__class__.__name__ == 'All3DDescriptors' or \
                featurizer.__class__.__name__ == 'Mol2Vec':
            scaler = MinMaxScaler()

    final_steps = [('standardizer', _get_standardizer(trial, featurizer)), ('featurizer', featurizer),
                   ('scaler', scaler),
                   ('feature_selector', feature_selector), ('model', sk_model)]
    return final_steps


def preset_keras_models(trial, data: Dataset) -> list:
    """
    Preset keras models for hyperparameter optimization.

    Parameters
    ----------
    trial: optuna.trial.Trial
        Optuna trial object.
    data: Dataset
        Dataset object.
    validation_dataset: Dataset
        Dataset object used for validation.

    Returns
    -------
    final_steps: list
        List of steps for keras models for hyperparameter optimization.
    """
    mode = data.mode
    n_classes = _get_n_classes(data)
    # TODO: in multitask when n_classes > 2 for different tasks, this will not work (LabelOneHotEncoder needs to work
    #  on different y columns)
    label_encoder = LabelOneHotEncoder() if mode == 'classification' and n_classes[0] > 2 else PassThroughTransformer()
    featurizer_type = trial.suggest_categorical('featurizer_type', ['1D', '2D'])
    featurizer = _get_featurizer(trial, featurizer_type)
    feature_selector = PassThroughTransformer()
    if featurizer_type == '1D':
        if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
                featurizer.__class__.__name__ == 'All3DDescriptors':
            scaler = _get_scaler(trial)
        else:
            scaler = PassThroughTransformer()
        input_shape = (len(featurizer.feature_names),)
        if featurizer.__class__.__name__ == 'nc_mfp':
            mode = data.mode
            multitask = True if data.n_tasks > 1 else False
            feature_selector = _get_feature_selector(trial, task_type=mode, multitask=multitask)
            while feature_selector.__class__.__name__ == 'PassThroughTransformer':
                feature_selector = _get_feature_selector(trial, task_type=mode, multitask=multitask)
    else:
        scaler = PassThroughTransformer()
        input_shape = featurizer.fit(data).shape
    keras_model = _get_keras_model(trial, input_shape, data)
    final_steps = [('label_encoder', label_encoder), ('standardizer', _get_standardizer(trial, featurizer)),
                   ('featurizer', featurizer), ('scaler', scaler), ('feature_selector', feature_selector),
                   ('model', keras_model)]
    return final_steps


def preset_all_models(trial, data: Dataset) -> list:
    """
    Preset all models for the trial.

    Parameters
    ----------
    trial: optuna.trial.Trial
        Optuna trial object.
    data: Dataset
        Dataset to be used for model selection.

    Returns
    -------
    final_steps: list
        List of steps for the pipeline.
    """
    model_type_choice = trial.suggest_categorical('model_type', ['keras', 'sklearn', 'deepchem'])
    if model_type_choice == 'keras':
        return preset_keras_models(trial, data)
    elif model_type_choice == 'sklearn':
        return preset_sklearn_models(trial, data)
    elif model_type_choice == 'deepchem':
        return preset_deepchem_models(trial, data)
    else:
        raise ValueError("Unknown model type: %s" % model_type_choice)


def prepare_dataset_for_textcnn(dataset: Dataset, max_length: int = 150, pad_char: str = 'E') -> Dataset:
    """
    Prepares a dataset for use with the TextCNN model.

    Parameters
    ----------
    dataset: Dataset
        Dataset to be prepared.
    max_length: int, optional (default 150)
        Maximum length of the SMILES strings.
    pad_char: str, optional (default 'E')
        Character to use for padding.

    Returns
    -------
    dataset: Dataset
        Prepared dataset.
    """
    padded_smiles = [smile.ljust(max_length, pad_char) for smile in dataset.smiles]
    truncated_smiles = [smile[:max_length] for smile in padded_smiles]
    dataset._X = truncated_smiles
    dataset._ids = truncated_smiles
    return dataset
