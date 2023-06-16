from optuna import Trial

from deepmol.models.keras_model_builders import keras_dense_model, keras_cnn_model, keras_tabular_transformer_model


def keras_dense_classification_step(trial: Trial, input_shape: tuple):
    input_dim = input_shape[0]
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 5)
    layers_units = [trial.suggest_int(f'layer_units_{i}', 8, 64) for i in range(n_hidden_layers + 1)]
    layers_units[-1] = 1
    dropouts = [trial.suggest_float(f'dropout_{i}', 0.0, 0.8) for i in range(n_hidden_layers + 1)]
    activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh']) for i in
                   range(n_hidden_layers + 1)]
    activations[-1] = 'sigmoid'
    batch_normalization = [trial.suggest_categorical(f'batch_normalization_{i}', [True, False]) for i in
                           range(n_hidden_layers + 1)]
    l1_l2 = [(trial.suggest_float(f'l1_{i}', 1e-6, 1e-2, log=True),
              trial.suggest_float(f'l2_{i}', 1e-6, 1e-2, log=True)) for i in range(n_hidden_layers)]
    # TODO: loss and metrics shoud be dynamic and metric the same as in the pipeline
    loss = 'binary_crossentropy'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metrics = ['accuracy']
    model_kwargs = {'input_dim': input_dim, 'n_hidden_layers': n_hidden_layers, 'layers_units': layers_units,
                    'dropouts': dropouts, 'activations': activations, 'batch_normalization': batch_normalization,
                    'l1_l2': l1_l2, 'loss': loss, 'optimizer': optimizer, 'metrics': metrics}
    return keras_dense_model(model_kwargs=model_kwargs)


def keras_cnn_classification_step(trial: Trial, input_shape: tuple):
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
    last_layer_activation = 'sigmoid'
    loss = 'binary_crossentropy'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metrics = ['accuracy']
    model_kwargs = {'input_dim': input_dim, 'g_noise': g_noise, 'n_conv_layers': n_conv_layers, 'filters': filters,
                    'kernel_sizes': kernel_sizes, 'strides': strides, 'activations': activations,
                    'dense_units': dense_units, 'dense_activation': dense_activation, 'dropout': dropout,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_layer_activation, 'loss': loss,
                    'optimizer': optimizer, 'metrics': metrics}
    print(model_kwargs)
    return keras_cnn_model(model_kwargs=model_kwargs)


def keras_tabular_transformer_step(trial: Trial, input_shape: tuple):
    input_dim = input_shape[0]
    embedding_output_dim = trial.suggest_int('embedding_output_dim', 8, 64)
    n_attention_layers = trial.suggest_int('n_attention_layers', 1, 4)
    n_heads = trial.suggest_int('n_heads', 2, 8)
    dropout_attention_l = trial.suggest_float('dropout_attention_l', 0.0, 0.5)
    dense_units = trial.suggest_int('dense_units', 32, 128)
    last_layer_units = 1
    last_layer_activation = 'sigmoid'
    loss = 'binary_crossentropy'
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    metric = 'accuracy'
    model_kwargs = {'input_dim': input_dim, 'embedding_output_dim': embedding_output_dim,
                    'n_attention_layers': n_attention_layers, 'n_heads': n_heads,
                    'dropout_attention_l': dropout_attention_l, 'dense_units': dense_units,
                    'last_layer_units': last_layer_units, 'last_layer_activation': last_layer_activation, 'loss': loss,
                    'optimizer': optimizer, 'metric': metric}
    return keras_tabular_transformer_model(model_kwargs=model_kwargs)


_TABULAR_CLASSIFICATION_MODELS = {'keras_dense': keras_dense_classification_step,
                                  'keras_cnn': keras_cnn_classification_step,
                                  'keras_tabular_transformer': keras_tabular_transformer_step
                                  }

_TABULAR_REGRESSION_MODELS = {  # 'keras_dense': keras_dense_regression_step,
}


def _get_keras_model(trial, task_type: str, input_shape: tuple):
    if task_type == 'classification':
        model_name = trial.suggest_categorical('model', ['keras_tabular_transformer'])
        return _TABULAR_CLASSIFICATION_MODELS[model_name](trial, input_shape)
    elif task_type == 'regression':
        model_name = trial.suggest_categorical('model', ['keras_dense'])
        return _TABULAR_REGRESSION_MODELS[model_name](trial, input_shape)
    else:
        raise ValueError(f'Unknown task type: {task_type}')
