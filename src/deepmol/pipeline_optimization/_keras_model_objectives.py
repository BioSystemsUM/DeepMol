from optuna import Trial
from tensorflow.keras.callbacks import EarlyStopping

from deepmol.datasets import Dataset
from deepmol.datasets._utils import _get_n_classes
from deepmol.models._utils import _get_last_layer_info_based_on_mode
from deepmol.models.keras_model_builders import *


def keras_fcnn_step(trial: Trial, input_shape: tuple, label_names: List[str] = None,
                    last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                    losses: Union[List[str], Dict[str, str]] = None, metrics: List[str] = None,
                    mode: Union[str, List[str]] = 'classification') -> KerasModel:
    """
    Create a fully connected neural network for a given trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    label_names : list of str, optional
        Names of the labels.
    last_layers_units : list of int, optional
        Number of units in the last layers.
    last_layers_activations : list of str, optional
        Activation functions of the last layers.
    losses : list of str or dict of str, optional
        Loss functions for each task.
    metrics : list of str, optional
        Metrics used for evaluating the model.
    mode : str or list of str, optional
        Mode of the model (classification or regression or list of both).

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    input_dim = input_shape[0]
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 5)
    hidden_units = [trial.suggest_int(f'hidden_layer_units_{i}', 8, 64) for i in range(n_hidden_layers)]
    hidden_dropouts = [trial.suggest_float(f'hidden_dropout_{i}', 0.0, 0.8) for i in range(n_hidden_layers)]
    hidden_activations = [trial.suggest_categorical(f'hidden_activation_{i}', ['relu', 'tanh']) for i in
                          range(n_hidden_layers)]
    batch_normalization = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                           range(n_hidden_layers)]
    l1_l2 = [(trial.suggest_float(f'l1_{i}', 1e-6, 1e-2, log=True),
              trial.suggest_float(f'l2_{i}', 1e-6, 1e-2, log=True)) for i in range(n_hidden_layers)]
    n_tasks = len(last_layers_units)
    model_kwargs = {'input_dim': input_dim, 'n_tasks': n_tasks, 'label_names': label_names,
                    'n_hidden_layers': n_hidden_layers, 'hidden_units': hidden_units,
                    'hidden_activations': hidden_activations, 'hidden_regularizers': l1_l2,
                    'hidden_dropouts': hidden_dropouts, 'batch_normalization': batch_normalization,
                    'last_layers_units': last_layers_units, 'last_layers_activations': last_layers_activations,
                    'optimizer': 'adam', 'losses': losses, 'metrics': metrics, 'callbacks': [EarlyStopping(patience=10)]}
    keras_kwargs = {'mode': mode, 'batch_size': trial.suggest_categorical('batch_size_fcnn', [16, 32, 64, 128, 256]),
                    'epochs': trial.suggest_categorical('epochs_fcnn', [30, 50, 100, 150, 200, 300, 500, 1000])}
    return keras_fcnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_1d_cnn_step(trial: Trial, input_shape: tuple, label_names: List[str] = None,
                      last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                      losses: Union[List[str], Dict[str, str]] = None, metrics: List[str] = None,
                      mode: Union[str, List[str]] = 'classification'):
    """
    Create a 1D convolutional neural network for a given trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    label_names : list of str, optional
        Names of the labels.
    last_layers_units : list of int, optional
        Number of units in the last layers.
    last_layers_activations : list of str, optional
        Activation functions of the last layers.
    losses : list of str or dict of str, optional
        Loss functions for each task.
    metrics : list of str, optional
        Metrics used for evaluating the model.
    mode : str or list of str, optional
        Mode of the model (classification or regression or list of both).

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    input_dim = input_shape[0]
    g_noise = trial.suggest_float('g_noise', 0.01, 0.1)
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
    filters = [trial.suggest_int(f'filter_{i}', 4, 32) for i in range(n_conv_layers)]
    kernel_sizes = [trial.suggest_int(f'kernel_size_{i}', 16, 64) for i in range(n_conv_layers)]
    strides = [trial.suggest_int(f'stride_{i}', 1, 2) for i in range(n_conv_layers)]
    conv_activations = [trial.suggest_categorical(f'conv_activation_{i}',
                                                  ['relu', 'tanh']) for i in range(n_conv_layers)]
    conv_dropouts = [trial.suggest_float(f'conv_dropout_{i}', 0.0, 0.8) for i in range(n_conv_layers)]
    conv_batch_norms = [trial.suggest_categorical(f'conv_batch_norm_{i}', [True, False]) for i in range(n_conv_layers)]
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    dense_dropout = trial.suggest_float('dropout', 0.0, 0.8)
    model_kwargs = {'input_dim': input_dim, 'n_tasks': len(last_layers_units), 'label_names': label_names,
                    'g_noise': g_noise, 'n_conv_layers': n_conv_layers, 'filters': filters,
                    'kernel_sizes': kernel_sizes, 'strides': strides, 'conv_activations': conv_activations,
                    'conv_dropouts': conv_dropouts, 'conv_batch_norms': conv_batch_norms, 'dense_units': dense_units,
                    'dense_activation': dense_activation, 'dense_dropout': dense_dropout,
                    'last_layers_units': last_layers_units, 'last_layers_activations': last_layers_activations,
                    'losses': losses, 'optimizer': 'adam', 'metrics': metrics, 'callbacks': [EarlyStopping(patience=10)]}
    keras_kwargs = {'mode': mode, 'batch_size': trial.suggest_categorical('batch_size_1d_conv', [16, 32, 64, 128, 256]),
                    'epochs': trial.suggest_categorical('epochs_1d_conv', [30, 50, 100, 150, 200, 300, 500, 1000])}
    return keras_1d_cnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_tabular_transformer_step(trial: Trial, input_shape: tuple, label_names: List[str] = None,
                                   last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                                   losses: Union[List[str], Dict[str, str]] = None, metrics: List[str] = None,
                                   mode: Union[str, List[str]] = 'classification'):
    """
    Create a tabular transformer model for a given trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    label_names : list of str, optional
        Names of the labels.
    last_layers_units : list of int, optional
        Number of units in the last layers.
    last_layers_activations : list of str, optional
        Activation functions of the last layers.
    losses : list of str or dict of str, optional
        Loss functions for each task.
    metrics : list of str, optional
        Metrics used for evaluating the model.
    mode : str or list of str, optional
        Mode of the model (classification or regression or list of both).

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    input_dim = input_shape[0]
    embedding_output_dim = trial.suggest_categorical('embedding_output_dim', [8, 16, 32, 64, 128])
    n_attention_layers = trial.suggest_categorical('n_attention_layers', [1, 2, 3, 4])
    n_attention_heads = trial.suggest_categorical('n_attention_heads', [1, 2, 4, 8])
    attention_dropouts = [trial.suggest_float(f'attention_dropout_{i}', 0.0, 0.5) for i in range(n_attention_layers)]
    attention_key_dims = [trial.suggest_categorical(f'attention_key_dim_{i}', [2, 4, 8, 16])
                          for i in range(n_attention_layers)]
    dense_units = trial.suggest_int('dense_units', 8, 128, step=8)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5)
    model_kwargs = {'input_dim': input_dim, 'label_names': label_names, 'n_tasks': len(last_layers_units),
                    'embedding_output_dim': embedding_output_dim, 'n_attention_layers': n_attention_layers,
                    'n_attention_heads': n_attention_heads, 'attention_dropouts': attention_dropouts,
                    'attention_key_dims': attention_key_dims, 'dense_units': dense_units,
                    'dense_activation': dense_activation, 'dense_dropout': dense_dropout, 'optimizer': 'adam',
                    'last_layers_units': last_layers_units, 'last_layers_activations': last_layers_activations,
                    'losses': losses, 'metrics': metrics, 'callbacks': [EarlyStopping(patience=10)]}
    keras_kwargs = {'mode': mode, 'batch_size': trial.suggest_categorical('batch_size_tabular_transformer', [16, 32, 64, 128, 256]),
                    'epochs': trial.suggest_categorical('epochs_tabular_transformer', [30, 50, 100, 150, 200, 300, 500, 1000])}
    return keras_tabular_transformer_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_simple_rnn_step(trial: Trial, input_shape: tuple, label_names: List[str] = None,
                          last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                          losses: Union[List[str], Dict[str, str]] = None, metrics: List[str] = None,
                          mode: Union[str, List[str]] = 'classification'):
    """
    Create a simple RNN model for a given trial.


    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    label_names : list of str, optional
        Names of the labels.
    last_layers_units : list of int, optional
        Number of units in the last layers.
    last_layers_activations : list of str, optional
        Activation functions of the last layers.
    losses : list of str or dict of str, optional
        Loss functions for each task.
    metrics : list of str, optional
        Metrics used for evaluating the model.
    mode : str or list of str, optional
        Mode of the model (classification or regression or list of both).

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    input_dim = input_shape
    n_rnn_layers = trial.suggest_int('n_rnn_layers', 1, 3)
    rnn_units = [trial.suggest_int(f'rnn_units_{i}', 32, 256) for i in range(n_rnn_layers)]
    rnn_dropouts = [trial.suggest_float(f'rnn_dropouts_{i}', 0.0, 0.5) for i in range(n_rnn_layers)]
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    model_kwargs = {'input_dim': input_dim, 'label_names': label_names, 'n_tasks': len(label_names),
                    'n_rnn_layers': n_rnn_layers, 'rnn_units': rnn_units, 'rnn_dropouts': rnn_dropouts,
                    'dense_units': dense_units, 'dense_dropout': dense_dropout, 'dense_activation': dense_activation,
                    'optimizer': 'adam', 'last_layers_units': last_layers_units,
                    'last_layers_activations': last_layers_activations, 'losses': losses, 'metrics': metrics,
                    'callbacks': [EarlyStopping(patience=10)]}
    keras_kwargs = {'mode': mode, 'batch_size': trial.suggest_categorical('batch_size_simple_rnn', [16, 32, 64, 128, 256]),
                    'epochs': trial.suggest_categorical('epochs_simple_rnn', [30, 50, 100, 150, 200, 300, 500, 1000])}
    return keras_simple_rnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_rnn_step(trial: Trial, input_shape: tuple, label_names: List[str] = None,
                   last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                   losses: Union[List[str], Dict[str, str]] = None, metrics: List[str] = None,
                   mode: Union[str, List[str]] = 'classification'):
    """
    Create a RNN model for a given trial.


    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    label_names : list of str, optional
        Names of the labels.
    last_layers_units : list of int, optional
        Number of units in the last layers.
    last_layers_activations : list of str, optional
        Activation functions of the last layers.
    losses : list of str or dict of str, optional
        Loss functions for each task.
    metrics : list of str, optional
        Metrics used for evaluating the model.
    mode : str or list of str, optional
        Mode of the model (classification or regression or list of both).

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    input_dim = input_shape
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
    lstm_units = [trial.suggest_int(f'lstm_units_{i}', 16, 256) for i in range(n_lstm_layers)]
    lstm_dropout = [trial.suggest_float(f'lstm_dropout_{i}', 0.0, 0.5) for i in range(n_lstm_layers)]
    n_gru_layers = trial.suggest_int('n_gru_layers', 1, 3)
    gru_units = [trial.suggest_int(f'gru_units_{i}', 16, 256) for i in range(n_gru_layers)]
    gru_dropout = [trial.suggest_float(f'gru_dropout_{i}', 0.0, 0.5) for i in range(n_gru_layers)]
    dense_units = trial.suggest_int('dense_units', 16, 256, step=16)
    dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    model_kwargs = {'input_dim': input_dim, 'label_names': label_names, 'n_tasks': len(label_names),
                    'n_lstm_layers': n_lstm_layers, 'lstm_units': lstm_units, 'lstm_dropout': lstm_dropout,
                    'n_gru_layers': n_gru_layers, 'gru_units': gru_units, 'gru_dropout': gru_dropout,
                    'dense_units': dense_units, 'dense_dropout': dense_dropout, 'dense_activation': dense_activation,
                    'optimizer': 'adam', 'last_layers_units': last_layers_units,
                    'last_layers_activations': last_layers_activations, 'losses': losses, 'metrics': metrics,
                    'callbacks': [EarlyStopping(patience=10)]}
    keras_kwargs = {'mode': mode, 'batch_size': trial.suggest_categorical('batch_size_rnn', [16, 32, 64, 128, 256]),
                    'epochs': trial.suggest_categorical('epochs_rnn', [30, 50, 100, 150, 200, 300, 500, 1000])}
    return keras_rnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_bidirectional_rnn_step(trial: Trial, input_shape: tuple, label_names: List[str] = None,
                                 last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                                 losses: Union[List[str], Dict[str, str]] = None, metrics: List[str] = None,
                                 mode: Union[str, List[str]] = 'classification'):
    """
    Create a bidirectional RNN model for a given trial.


    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    label_names : list of str, optional
        Names of the labels.
    last_layers_units : list of int, optional
        Number of units in the last layers.
    last_layers_activations : list of str, optional
        Activation functions of the last layers.
    losses : list of str or dict of str, optional
        Loss functions for each task.
    metrics : list of str, optional
        Metrics used for evaluating the model.
    mode : str or list of str, optional
        Mode of the model (classification or regression or list of both).

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    input_dim = input_shape
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
    lstm_units = [trial.suggest_int(f'lstm_units_{i}', 16, 256) for i in range(n_lstm_layers)]
    lstm_dropout = [trial.suggest_float(f'lstm_dropout_{i}', 0.0, 0.5) for i in range(n_lstm_layers)]
    n_gru_layers = trial.suggest_int('n_gru_layers', 1, 3)
    gru_units = [trial.suggest_int(f'gru_units_{i}', 16, 256) for i in range(n_gru_layers)]
    gru_dropout = [trial.suggest_float(f'gru_dropout_{i}', 0.0, 0.5) for i in range(n_gru_layers)]
    dense_units = trial.suggest_int('dense_units', 16, 256, step=16)
    dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    model_kwargs = {'input_dim': input_dim, 'label_names': label_names, 'n_tasks': len(label_names),
                    'n_lstm_layers': n_lstm_layers, 'lstm_units': lstm_units, 'lstm_dropout': lstm_dropout,
                    'n_gru_layers': n_gru_layers, 'gru_units': gru_units, 'gru_dropout': gru_dropout,
                    'dense_units': dense_units, 'dense_dropout': dense_dropout, 'dense_activation': dense_activation,
                    'optimizer': 'adam', 'last_layers_units': last_layers_units,
                    'last_layers_activations': last_layers_activations, 'losses': losses, 'metrics': metrics,
                    'callbacks': [EarlyStopping(patience=10)]}
    keras_kwargs = {'mode': mode, 'batch_size': trial.suggest_categorical('batch_size_bi_rnn', [16, 32, 64, 128, 256]),
                    'epochs': trial.suggest_categorical('epochs_bi_rnn', [30, 50, 100, 150, 200, 300, 500, 1000])}
    return keras_bidirectional_rnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


_TABULAR_KERAS_MODELS = {'keras_dense': keras_fcnn_step,
                         'keras_1d_cnn': keras_1d_cnn_step,
                         'keras_tabular_transformer': keras_tabular_transformer_step,
                         }

_2D_KERAS_MODELS = {'keras_simple_rnn': keras_simple_rnn_step,
                    'keras_rnn': keras_rnn_step,
                    'keras_bidirectional_rnn': keras_bidirectional_rnn_step,
                    }


def _get_keras_model(trial, input_shape: tuple, dataset: Dataset):
    """
    Get the Keras model based on the trial and the dataset.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object that stores the hyperparameters.
    input_shape : tuple
        Shape of the input data.
    dataset : Dataset
        Dataset object.

    Returns
    -------
    KerasModel
        KerasModel object.
    """
    mode = dataset.mode
    label_names = dataset.label_names
    n_classes = _get_n_classes(dataset)
    if isinstance(mode, str):
        loss, last_layer_activations, last_layer_units = _get_last_layer_info_based_on_mode(mode, n_classes[0])
        metric = ['accuracy'] if mode == 'classification' else ['mean_squared_error']
    elif isinstance(mode, list):
        unique_mode = set(mode)
        if len(unique_mode) > 1:
            metric = {label_names[i]: ['accuracy'] if mode[i] == 'classification' else ['mean_squared_error'] for i in
                      range(len(mode))}
        else:
            metric = ['accuracy'] if mode[0] == 'classification' else ['mean_squared_error']
        loss = []
        last_layer_activations = []
        last_layer_units = []
        for i, m in enumerate(mode):
            m_loss, m_last_layer_activations, m_last_layer_units = _get_last_layer_info_based_on_mode(m, n_classes[i])
            loss.extend(m_loss)
            last_layer_activations.extend(m_last_layer_activations)
            last_layer_units.extend(m_last_layer_units)
    else:
        raise ValueError(f'Unknown mode {mode}')
    if len(input_shape) == 1:  # tabular data
        model_name = trial.suggest_categorical('1d_model', list(_TABULAR_KERAS_MODELS.keys()))
        return _TABULAR_KERAS_MODELS[model_name](trial, input_shape=input_shape, label_names=label_names,
                                                 last_layers_units=last_layer_units,
                                                 last_layers_activations=last_layer_activations,
                                                 losses=loss, metrics=metric, mode=mode)
    elif len(input_shape) == 2:
        model_name = trial.suggest_categorical('2d_model', list(_2D_KERAS_MODELS.keys()))
        return _2D_KERAS_MODELS[model_name](trial, input_shape=input_shape, label_names=label_names,
                                            last_layers_units=last_layer_units,
                                            last_layers_activations=last_layer_activations,
                                            losses=loss, metrics=metric, mode=mode)
    else:
        raise ValueError(f'Input shape {input_shape} not supported')
