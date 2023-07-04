from optuna import Trial

from deepmol.models.keras_model_builders import *


def keras_dense_classification_step(trial: Trial, input_shape: tuple, last_layer_units: int = 1,
                                    last_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                                    metrics: list = ['accuracy']):
    """
    Optuna model step for a dense Keras model for classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.
    last_layer_units : int, optional
        The number of units of the last layer, by default 1.
    last_activation : str, optional
        The activation function of the last layer, by default 'sigmoid'.
    loss : str, optional
        The loss function, by default 'binary_crossentropy'.
    metrics : list, optional
        The metrics, by default ['accuracy'].

    Returns
    -------
    keras_dense_model
        A dense Keras model.
    """
    input_dim = input_shape[0]
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 5)
    layers_units = [trial.suggest_int(f'layer_units_{i}', 8, 64) for i in range(n_hidden_layers + 1)]
    layers_units[-1] = last_layer_units
    dropouts = [trial.suggest_float(f'dropout_{i}', 0.0, 0.8) for i in range(n_hidden_layers + 1)]
    activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh']) for i in
                   range(n_hidden_layers + 1)]
    activations[-1] = last_activation
    batch_normalization = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                           range(n_hidden_layers + 1)]
    l1_l2 = [(trial.suggest_float(f'l1_{i}', 1e-6, 1e-2, log=True),
              trial.suggest_float(f'l2_{i}', 1e-6, 1e-2, log=True)) for i in range(n_hidden_layers)]
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    model_kwargs = {'input_dim': input_dim, 'n_hidden_layers': n_hidden_layers, 'layers_units': layers_units,
                    'dropouts': dropouts, 'activations': activations, 'batch_normalization': batch_normalization,
                    'l1_l2': l1_l2, 'loss': loss, 'optimizer': optimizer, 'metrics': metrics}
    return keras_dense_model(model_kwargs=model_kwargs)


def keras_dense_regression_step(trial: Trial, input_shape: tuple):
    """
    Optuna model step for a dense Keras model for regression.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_dense_model
        A dense Keras model.
    """
    input_dim = input_shape[0]
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 5)
    layers_units = [trial.suggest_int(f'layer_units_{i}', 8, 64) for i in range(n_hidden_layers + 1)]
    layers_units[-1] = 1
    dropouts = [trial.suggest_float(f'dropout_{i}', 0.0, 0.8) for i in range(n_hidden_layers + 1)]
    activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh']) for i in
                   range(n_hidden_layers + 1)]
    activations[-1] = 'linear'
    batch_normalization = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                           range(n_hidden_layers + 1)]
    l1_l2 = [(trial.suggest_float(f'l1_{i}', 1e-6, 1e-2, log=True),
              trial.suggest_float(f'l2_{i}', 1e-6, 1e-2, log=True)) for i in range(n_hidden_layers)]
    # TODO: loss and metrics shoud be dynamic and metric the same as in the pipeline
    loss = 'mean_squared_error'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metrics = ['mse']
    model_kwargs = {'input_dim': input_dim, 'n_hidden_layers': n_hidden_layers, 'layers_units': layers_units,
                    'dropouts': dropouts, 'activations': activations, 'batch_normalization': batch_normalization,
                    'l1_l2': l1_l2, 'loss': loss, 'optimizer': optimizer, 'metrics': metrics}
    keras_kwargs = {'mode': 'regression'}
    return keras_dense_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_1D_cnn_classification_step(trial: Trial, input_shape: tuple, last_layer_units: int = 1,
                                        last_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                                        metrics: list = ['accuracy']):
    """
    Optuna model step for a 1D CNN Keras model for classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.
    last_layer_units : int, optional
        The number of units of the last layer, by default 1.
    last_activation : str, optional
        The activation function of the last layer, by default 'sigmoid'.
    loss : str, optional
        The loss function, by default 'binary_crossentropy'.
    metrics : list, optional
        The metrics, by default ['accuracy'].

    Returns
    -------
    keras_1D_cnn_model
        A 1D CNN Keras model.
    """
    input_dim = input_shape[0]
    g_noise = trial.suggest_float('g_noise', 0.01, 0.1)
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
    filters = [trial.suggest_int(f'filter_{i}', 4, 32) for i in range(n_conv_layers)]
    kernel_sizes = [trial.suggest_int(f'kernel_size_{i}', 16, 64) for i in range(n_conv_layers)]
    strides = [trial.suggest_int(f'stride_{i}', 1, 2) for i in range(n_conv_layers)]
    activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh']) for i in range(n_conv_layers)]
    dense_units = trial.suggest_int('dense_units', 32, 256)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    model_kwargs = {'input_dim': input_dim, 'g_noise': g_noise, 'n_conv_layers': n_conv_layers, 'filters': filters,
                    'kernel_sizes': kernel_sizes, 'strides': strides, 'activations': activations,
                    'dense_units': dense_units, 'dense_activation': dense_activation, 'dropout': dropout,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_activation, 'loss': loss,
                    'optimizer': optimizer, 'metrics': metrics}
    return keras_1D_cnn_model(model_kwargs=model_kwargs)


def keras_1D_cnn_regression_step(trial: Trial, input_shape: tuple):
    """
    Optuna model step for a 1D CNN Keras model for regression.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_1D_cnn_model
        A 1D CNN Keras model.
    """
    input_dim = input_shape[0]
    g_noise = trial.suggest_float('g_noise', 0.01, 0.1)
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
    filters = [trial.suggest_int(f'filter_{i}', 4, 32) for i in range(n_conv_layers)]
    kernel_sizes = [trial.suggest_int(f'kernel_size_{i}', 16, 64) for i in range(n_conv_layers)]
    strides = [trial.suggest_int(f'stride_{i}', 1, 2) for i in range(n_conv_layers)]
    activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh']) for i in range(n_conv_layers)]
    dense_units = trial.suggest_int('dense_units', 32, 256)
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    last_layer_units = 1
    last_layer_activation = 'linear'
    loss = 'mean_squared_error'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metrics = ['mse']
    model_kwargs = {'input_dim': input_dim, 'g_noise': g_noise, 'n_conv_layers': n_conv_layers, 'filters': filters,
                    'kernel_sizes': kernel_sizes, 'strides': strides, 'activations': activations,
                    'dense_units': dense_units, 'dense_activation': dense_activation, 'dropout': dropout,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_layer_activation, 'loss': loss,
                    'optimizer': optimizer, 'metrics': metrics}
    keras_kwargs = {'mode': 'regression'}
    return keras_1D_cnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_tabular_transformer_classification_step(trial: Trial, input_shape: tuple, last_layer_units: int = 1,
                                                  last_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                                                  metrics: list = ['accuracy']):
    """
    Optuna model step for a Tabular Transformer Keras model for classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.
    last_layer_units : int, optional
        The number of units of the last layer, by default 1.
    last_activation : str, optional
        The activation function of the last layer, by default 'sigmoid'.
    loss : str, optional
        The loss function, by default 'binary_crossentropy'.
    metrics : list, optional
        The metrics, by default ['accuracy'].

    Returns
    -------
    keras_tabular_transformer_model
        A Tabular Transformer Keras model.
    """
    input_dim = input_shape[0]
    embedding_output_dim = trial.suggest_int('embedding_output_dim', 8, 64)
    n_attention_layers = trial.suggest_int('n_attention_layers', 1, 4)
    n_heads = trial.suggest_int('n_heads', 2, 8)
    dropout_attention_l = trial.suggest_float('dropout_attention_l', 0.0, 0.5)
    dense_units = trial.suggest_int('dense_units', 32, 128)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'accuracy'
    model_kwargs = {'input_dim': input_dim, 'embedding_output_dim': embedding_output_dim,
                    'n_attention_layers': n_attention_layers, 'n_heads': n_heads,
                    'dropout_attention_l': dropout_attention_l, 'dense_units': dense_units,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metric}
    return keras_tabular_transformer_model(model_kwargs=model_kwargs)


# TODO: generally fails (predict returns nan)
def keras_tabular_transformer_regression_step(trial: Trial, input_shape: tuple):
    """
    Optuna model step for a Tabular Transformer Keras model for regression.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_tabular_transformer_model
        A Tabular Transformer Keras model.
    """
    input_dim = input_shape[0]
    embedding_output_dim = trial.suggest_int('embedding_output_dim', 8, 64)
    n_attention_layers = trial.suggest_int('n_attention_layers', 1, 4)
    n_heads = trial.suggest_int('n_heads', 2, 8)
    dropout_attention_l = trial.suggest_float('dropout_attention_l', 0.0, 0.5)
    dense_units = trial.suggest_int('dense_units', 32, 128)
    last_layer_units = 1
    last_layer_activation = 'linear'
    loss = 'mean_squared_error'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'mse'
    model_kwargs = {'input_dim': input_dim, 'embedding_output_dim': embedding_output_dim,
                    'n_attention_layers': n_attention_layers, 'n_heads': n_heads,
                    'dropout_attention_l': dropout_attention_l, 'dense_units': dense_units,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_layer_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metric}
    keras_kwargs = {'mode': 'regression'}
    return keras_tabular_transformer_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_simple_rnn_classification_step(trial: Trial, input_shape: tuple, last_layer_units: int = 1,
                                            last_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                                            metrics: list = ['accuracy']):
    """
    Optuna model step for a Simple RNN Keras model for classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.
    last_layer_units : int, optional
        The number of units of the last layer, by default 1.
    last_activation : str, optional
        The activation function of the last layer, by default 'sigmoid'.
    loss : str, optional
        The loss function, by default 'binary_crossentropy'.
    metrics : list, optional
        The metrics, by default ['accuracy'].

    Returns
    -------
    keras_simple_rnn_model
        A Simple RNN Keras model.
    """
    n_rnn_layers = trial.suggest_int('n_rnn_layers', 1, 3)
    rnn_units = trial.suggest_int('rnn_units', 32, 256, step=32)
    dropout_rnn = trial.suggest_float('dropout_rnn', 0.0, 0.8, step=0.2)
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5, step=0.1)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'accuracy'
    model_kwargs = {'input_dim': input_shape, 'n_rnn_layers': n_rnn_layers, 'rnn_units': rnn_units,
                    'dropout_rnn': dropout_rnn, 'dense_units': dense_units, 'dense_dropout': dense_dropout,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metric}
    return keras_simple_rnn_model(model_kwargs=model_kwargs)


def keras_simple_rnn_regression_step(trial: Trial, input_shape: tuple):
    """
    Optuna model step for a Simple RNN Keras model for regression.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_simple_rnn_model
        A Simple RNN Keras model.
    """
    n_rnn_layers = trial.suggest_int('n_rnn_layers', 1, 3)
    rnn_units = trial.suggest_int('rnn_units', 32, 256, step=32)
    dropout_rnn = trial.suggest_float('dropout_rnn', 0.0, 0.8, step=0.2)
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    dense_dropout = trial.suggest_float('dense_dropout', 0.0, 0.5, step=0.1)
    last_layer_units = 1
    last_layer_activation = 'linear'
    loss = 'mean_squared_error'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'mse'
    model_kwargs = {'input_dim': input_shape, 'n_rnn_layers': n_rnn_layers, 'rnn_units': rnn_units,
                    'dropout_rnn': dropout_rnn, 'dense_units': dense_units, 'dense_dropout': dense_dropout,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_layer_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metric}
    keras_kwargs = {'mode': 'regression'}
    return keras_simple_rnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_rnn_classification_step(trial: Trial, input_shape: tuple, last_layer_units: int = 1,
                                    last_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                                    metrics: list = ['accuracy']):
    """
    Optuna model step for a RNN Keras model for classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.
    last_layer_units : int, optional
        The number of units of the last layer, by default 1.
    last_activation : str, optional
        The activation function of the last layer, by default 'sigmoid'.
    loss : str, optional
        The loss function, by default 'binary_crossentropy'.
    metrics : list, optional
        The metrics, by default ['accuracy'].

    Returns
    -------
    keras_rnn_model
        A RNN Keras model.
    """
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    lstm_units = []
    for i in range(num_lstm_layers):
        lstm_units.append(trial.suggest_int(f'lstm_units_{i}', 32, 256, step=32))
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
    gru_units = []
    for i in range(num_gru_layers):
        gru_units.append(trial.suggest_int(f'gru_units_{i}', 32, 256, step=32))
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    model_kwargs = {'num_lstm_layers': num_lstm_layers, 'lstm_units': lstm_units, 'num_gru_layers': num_gru_layers,
                    'gru_units': gru_units, 'dense_units': dense_units, 'last_layer_units': last_layer_units,
                    'last_layer_activation': last_activation, 'loss': loss, 'optimizer': optimizer,
                    'metric': metrics}
    return keras_rnn_model(model_kwargs=model_kwargs)


def keras_rnn_regression_step(trial: Trial, input_shape: tuple):
    """
    Optuna model step for a RNN Keras model for regression.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_rnn_model
        A RNN Keras model.
    """
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    lstm_units = []
    for i in range(num_lstm_layers):
        lstm_units.append(trial.suggest_int(f'lstm_units_{i}', 32, 256, step=32))
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
    gru_units = []
    for i in range(num_gru_layers):
        gru_units.append(trial.suggest_int(f'gru_units_{i}', 32, 256, step=32))
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    last_layer_units = 1
    last_layer_activation = 'linear'
    loss = 'mean_squared_error'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'mse'
    model_kwargs = {'num_lstm_layers': num_lstm_layers, 'lstm_units': lstm_units, 'num_gru_layers': num_gru_layers,
                    'gru_units': gru_units, 'dense_units': dense_units, 'last_layer_units': last_layer_units,
                    'last_layer_activation': last_layer_activation, 'loss': loss, 'optimizer': optimizer,
                    'metric': metric}
    keras_kwargs = {'mode': 'regression'}
    return keras_rnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


def keras_bidirectional_rnn_classification_step(trial: Trial, input_shape: tuple, last_layer_units: int = 1,
                                                last_activation: str = 'sigmoid',
                                                loss: str = 'binary_crossentropy',
                                                metrics: list = ['accuracy']):
    """
    Optuna model step for a bidirectional RNN Keras model for classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.
    last_layer_units : int, optional
        The number of units of the last layer, by default 1.
    last_activation : str, optional
        The activation function of the last layer, by default 'sigmoid'.
    loss : str, optional
        The loss function, by default 'binary_crossentropy'.
    metrics : list, optional
        The metrics, by default ['accuracy'].

    Returns
    -------
    keras_bidirectional_rnn_model
        A bidirectional RNN Keras model.
    """
    input_dim = input_shape
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    lstm_units = []
    for i in range(num_lstm_layers):
        lstm_units.append(trial.suggest_int(f'lstm_units_{i}', 32, 256, step=32))
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
    gru_units = []
    for i in range(num_gru_layers):
        gru_units.append(trial.suggest_int(f'gru_units_{i}', 32, 256, step=32))
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    model_kwargs = {'input_dim': input_dim, 'num_lstm_layers': num_lstm_layers, 'lstm_units': lstm_units,
                    'num_gru_layers': num_gru_layers, 'gru_units': gru_units, 'dense_units': dense_units,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metrics}
    return keras_bidirectional_rnn_model(model_kwargs=model_kwargs)


def keras_bidirectional_rnn_regression_step(trial: Trial, input_shape: tuple):
    """
    Optuna model step for a bidirectional RNN Keras model for regression.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_bidirectional_rnn_model
        A bidirectional RNN Keras model.
    """
    input_dim = input_shape
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    lstm_units = []
    for i in range(num_lstm_layers):
        lstm_units.append(trial.suggest_int(f'lstm_units_{i}', 32, 256, step=32))
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
    gru_units = []
    for i in range(num_gru_layers):
        gru_units.append(trial.suggest_int(f'gru_units_{i}', 32, 256, step=32))
    dense_units = trial.suggest_int('dense_units', 32, 256, step=32)
    last_layer_units = 1
    last_layer_activation = 'linear'
    loss = 'mean_squared_error'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'mse'
    model_kwargs = {'input_dim': input_dim, 'num_lstm_layers': num_lstm_layers, 'lstm_units': lstm_units,
                    'num_gru_layers': num_gru_layers, 'gru_units': gru_units, 'dense_units': dense_units,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_layer_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metric}
    keras_kwargs = {'mode': 'regression'}
    return keras_bidirectional_rnn_model(model_kwargs=model_kwargs, keras_kwargs=keras_kwargs)


_TABULAR_CLASSIFICATION_MODELS = {'keras_dense': keras_dense_classification_step,
                                  'keras_1D_cnn': keras_1D_cnn_classification_step,
                                  'keras_tabular_transformer': keras_tabular_transformer_classification_step,
                                  }

_2D_CLASSIFICATION_MODELS = {'keras_simple_rnn': keras_simple_rnn_classification_step,
                             'keras_rnn': keras_rnn_classification_step,
                             'keras_bidirectional_rnn': keras_bidirectional_rnn_classification_step,
                             }

_TABULAR_REGRESSION_MODELS = {'keras_dense': keras_dense_regression_step,
                              'keras_1D_cnn': keras_1D_cnn_regression_step,
                              'keras_tabular_transformer': keras_tabular_transformer_regression_step,
                              }

_2D_REGRESSION_MODELS = {'keras_simple_rnn': keras_simple_rnn_regression_step,
                         'keras_rnn': keras_rnn_regression_step,
                         'keras_bidirectional_rnn': keras_bidirectional_rnn_regression_step,
                         }


def _get_keras_model(trial, task_type: str, n_classes: int, featurizer_type: str, input_shape: tuple):
    """
    Get a Keras model step for Optuna based on the task type and featurizer type.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.
    task_type : str
        The type of task, either classification or regression.
    n_classes : int
        The number of classes (only used for classification).
    featurizer_type : str
        The type of featurizer, either 1D or 2D.
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    keras_model
        A Keras model step for Optuna.
    """
    if task_type == 'classification':
        if n_classes > 2:
            loss = 'categorical_crossentropy'
            last_layer_activation = 'softmax'
            last_layer_units = n_classes
        else:
            loss = 'binary_crossentropy'
            last_layer_activation = 'sigmoid'
            last_layer_units = 1
        if featurizer_type == '1D':
            model_name = trial.suggest_categorical('1d_model', list(_TABULAR_CLASSIFICATION_MODELS.keys()))
            return _TABULAR_CLASSIFICATION_MODELS[model_name](trial, input_shape, last_layer_units=last_layer_units,
                                                              last_activation=last_layer_activation, loss=loss)
        elif featurizer_type == '2D':
            model_name = trial.suggest_categorical('2d_model', list(_2D_CLASSIFICATION_MODELS.keys()))
            return _2D_CLASSIFICATION_MODELS[model_name](trial, input_shape, last_layer_units=last_layer_units,
                                                         last_activation=last_layer_activation, loss=loss)
    elif task_type == 'regression':
        if featurizer_type == '1D':
            model_name = trial.suggest_categorical('1d_model', list(_TABULAR_REGRESSION_MODELS.keys()))
            return _TABULAR_REGRESSION_MODELS[model_name](trial, input_shape)
        elif featurizer_type == '2D':
            model_name = trial.suggest_categorical('2d_model', list(_2D_REGRESSION_MODELS.keys()))
            return _2D_REGRESSION_MODELS[model_name](trial, input_shape)
    else:
        raise ValueError(f'Unknown task type: {task_type}')
