from typing import List, Union, Tuple, Dict

from tensorflow.keras import regularizers, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, Reshape, Conv1D, Flatten, \
    Embedding, MultiHeadAttention, LayerNormalization, Input, SimpleRNN, LSTM, GRU, Bidirectional

from deepmol.models import KerasModel


def keras_fcnn_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None, n_hidden_layers: int = 1,
                             hidden_units: List[int] = None, hidden_activations: List[str] = None,
                             hidden_regularizers: List[Tuple[float, float]] = None,
                             hidden_dropouts: List[float] = None, batch_normalization: List[bool] = None,
                             last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                             optimizer: str = 'adam', losses: Union[List[str], Dict[str, str]] = None,
                             metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    """
    Build a fully connected neural network model using Keras.

    Parameters
    ----------
    input_dim : int
        Dimension of the input layer.
    n_tasks : int
        Number of tasks.
    label_names : list of str
        Names of the labels.
    n_hidden_layers : int
        Number of hidden layers.
    hidden_units : list of int
        Number of units in each hidden layer.
    hidden_activations : list of str
        Activation functions of each hidden layer.
    hidden_regularizers : list of tuple of float
        Regularizers of each hidden layer.
    hidden_dropouts : list of float
        Dropout rates of each hidden layer.
    batch_normalization : list of bool
        Whether to use batch normalization in each hidden layer.
    last_layers_units : list of int
        Number of units in each last layer.
    last_layers_activations : list of str
        Activation functions of each last layer.
    optimizer : str
        Optimizer.
    losses : list of str or dict of str
        Loss functions.
    metrics : list of str or dict of str
        Metrics.

    Returns
    -------
    model : keras.Model
        The built model.
    """
    label_names = label_names if label_names is not None else [f'label_{i}' for i in range(n_tasks)]
    hidden_units = hidden_units if hidden_units is not None else [64]
    hidden_activations = hidden_activations if hidden_activations is not None else ['relu']
    hidden_dropouts = hidden_dropouts if hidden_dropouts is not None else [0.0]
    batch_normalization = batch_normalization if batch_normalization is not None else [False]
    hidden_regularizers = hidden_regularizers if hidden_regularizers is not None else [(0.0, 0.0)]
    last_layers_units = last_layers_units if last_layers_units is not None else [1]
    last_layers_activations = last_layers_activations if last_layers_activations is not None else ['sigmoid']
    metrics = metrics if metrics is not None else ['accuracy']
    losses = losses if losses is not None else ['binary_crossentropy']
    assert n_hidden_layers == len(hidden_units) == len(hidden_activations) == len(hidden_dropouts) \
           == len(batch_normalization) == len(hidden_regularizers)
    assert n_tasks == len(last_layers_units) == len(last_layers_activations) == len(label_names)
    # Input layer
    input_layer = Input(shape=(input_dim,))
    shared_layers = input_layer
    # Hidden layers
    for i in range(n_hidden_layers):
        shared_layers = Dense(hidden_units[i], activation=hidden_activations[i],
                              kernel_regularizer=regularizers.l1_l2(*hidden_regularizers[i]))(shared_layers)
        shared_layers = Dropout(hidden_dropouts[i])(shared_layers)
        if batch_normalization[i]:
            shared_layers = BatchNormalization()(shared_layers)

    # Output layers
    outputs = []
    for i in range(n_tasks):
        task_layer = Dense(hidden_units[-1] / 2, activation=hidden_activations[-1],
                           kernel_regularizer=regularizers.l1_l2(*hidden_regularizers[-1]))(shared_layers)
        task_layer = Dropout(hidden_dropouts[-1])(task_layer)
        if batch_normalization[-1]:
            task_layer = BatchNormalization()(task_layer)

        task_output = Dense(last_layers_units[i], activation=last_layers_activations[i],
                            name=label_names[i])(task_layer)
        outputs.append(task_output)

    # Create model
    model = Model(inputs=input_layer, outputs=outputs)

    # Compile model
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def keras_fcnn_model(model_kwargs: dict = None,
                     keras_kwargs: dict = None) -> KerasModel:
    """
    Build a fully connected neural network model using Keras.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the Keras model.

    Returns
    -------
    model : KerasModel
        The built model.
    """
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = keras_kwargs.get('mode', 'classification')
    epochs = keras_kwargs.get('epochs', 150)
    batch_size = keras_kwargs.get('batch_size', 10)
    verbose = keras_kwargs.get('verbose', 0)
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=keras_fcnn_model_builder, mode=mode, epochs=epochs,
                      batch_size=batch_size, verbose=verbose, **model_kwargs)


def keras_1d_cnn_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None, g_noise: float = 0.05,
                               n_conv_layers: int = 2, filters: List[int] = None, kernel_sizes: List[int] = None,
                               strides: List[int] = None, conv_activations: List[str] = None,
                               conv_dropouts: List[float] = None, conv_batch_norms: List[bool] = None,
                               padding: str = 'same', dense_units: int = 128, dense_activation: str = 'relu',
                               dense_dropout: float = 0.5, last_layers_units: List[int] = None,
                               last_layers_activations: List[str] = None, optimizer: str = 'adam',
                               losses: Union[List[str], Dict[str, str]] = None,
                               metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    """
    Build a 1D convolutional neural network model using Keras.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    n_tasks : int
        Number of tasks.
    label_names : List[str]
        List of label names.
    g_noise : float
        Gaussian noise.
    n_conv_layers : int
        Number of convolutional layers.
    filters : List[int]
        List of filters.
    kernel_sizes : List[int]
        List of kernel sizes.
    strides : List[int]
        List of strides.
    conv_activations : List[str]
        List of convolutional activations.
    conv_dropouts : List[float]
        List of convolutional dropouts.
    conv_batch_norms : List[bool]
        List of convolutional batch normalizations.
    padding : str
        Padding.
    dense_units : int
        Number of dense units.
    dense_activation : str
        Dense activation.
    dense_dropout : float
        Dense dropout.
    last_layers_units : List[int]
        List of last layers units.
    last_layers_activations : List[str]
        List of last layers activations.
    optimizer : str
        Optimizer.
    losses : Union[List[str], Dict[str, str]]
        Losses.
    metrics : Union[List[str], Dict[str, str]]
        Metrics.

    Returns
    -------
    model : Model
        The built model.
    """
    label_names = label_names if label_names is not None else [f'label_{i}' for i in range(n_tasks)]
    filters = filters if filters is not None else [8, 16]
    kernel_sizes = kernel_sizes if kernel_sizes is not None else [32, 32]
    strides = strides if strides is not None else [1, 1]
    conv_activations = conv_activations if conv_activations is not None else ['relu', 'relu']
    conv_dropouts = conv_dropouts if conv_dropouts is not None else [0.1, 0.1]
    conv_batch_norms = conv_batch_norms if conv_batch_norms is not None else [True, True]
    last_layers_units = last_layers_units if last_layers_units is not None else [1]
    last_layers_activations = last_layers_activations if last_layers_activations is not None else ['sigmoid']
    metrics = metrics if metrics is not None else ['accuracy']
    losses = losses if losses is not None else ['binary_crossentropy']
    assert n_conv_layers == len(filters) == len(kernel_sizes) == len(conv_activations) == len(conv_dropouts) == len(
        conv_batch_norms)
    assert n_tasks == len(last_layers_units) == len(last_layers_activations) == len(label_names) == len(losses)

    # Input layer
    input_layer = Input(shape=(input_dim,))
    shared_layers = input_layer

    # Add Gaussian noise
    shared_layers = GaussianNoise(g_noise)(shared_layers)
    # Reshape tensor
    shared_layers = Reshape((input_dim, 1))(shared_layers)
    # Conv layers
    for i in range(n_conv_layers):
        shared_layers = Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], activation=conv_activations[i],
                               strides=strides[i], padding=padding)(shared_layers)
        shared_layers = Dropout(conv_dropouts[i])(shared_layers)
        if conv_batch_norms[i]:
            shared_layers = BatchNormalization()(shared_layers)
    # Flatten tensor
    shared_layers = Flatten()(shared_layers)
    # Output layers (1 dense + 1 final dense per task)
    output_layers = []
    for i in range(n_tasks):
        task_layers = Dense(dense_units, activation=dense_activation)(shared_layers)
        task_layers = Dropout(dense_dropout)(task_layers)
        task_layers = Dense(last_layers_units[i], activation=last_layers_activations[i],
                            name=label_names[i])(task_layers)
        output_layers.append(task_layers)

    # Build model
    model = Model(inputs=input_layer, outputs=output_layers)

    # Compile model
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def keras_1d_cnn_model(model_kwargs: dict = None,
                       keras_kwargs: dict = None) -> KerasModel:
    """
    Build a 1D convolutional neural network model using Keras.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the Keras model.

    Returns
    -------
    model : KerasModel
        The built model.
    """
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    keras_kwargs = keras_kwargs if keras_kwargs is not None else {}
    mode = keras_kwargs.get('mode', 'classification')
    epochs = keras_kwargs.get('epochs', 150)
    batch_size = keras_kwargs.get('batch_size', 10)
    verbose = keras_kwargs.get('verbose', 0)
    return KerasModel(model_builder=keras_1d_cnn_model_builder, mode=mode, epochs=epochs,
                      batch_size=batch_size, verbose=verbose, **model_kwargs)


def keras_tabular_transformer_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None,
                                            embedding_output_dim: int = 32, n_attention_layers: int = 2,
                                            n_attention_heads: int = 4, attention_dropouts: List[float] = None,
                                            attention_key_dims: List[int] = None, dense_units: int = 64,
                                            dense_activation: str = 'relu', dense_dropout: float = 0.1,
                                            last_layers_units: List[int] = None,
                                            last_layers_activations: List[str] = None, optimizer: str = 'adam',
                                            losses: Union[List[str], Dict[str, str]] = None,
                                            metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    """
    Build a transformer model using Keras.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    n_tasks : int
        Number of tasks.
    label_names : List[str]
        List of label names.
    embedding_output_dim : int
        Output dimension of the embedding layer.
    n_attention_layers : int
        Number of attention layers.
    n_attention_heads : int
        Number of attention heads.
    attention_dropouts : List[float]
        List of attention dropouts.
    attention_key_dims : List[int]
        List of attention key dimensions.
    dense_units : int
        Number of units in the dense layer.
    dense_activation : str
        Activation function for the dense layer.
    dense_dropout : float
        Dropout rate for the dense layer.
    last_layers_units : List[int]
        List of units in the last layers.
    last_layers_activations : List[str]
        List of activation functions for the last layers.
    optimizer : str
        Optimizer.
    losses : Union[List[str], Dict[str, str]]
        List of losses or dictionary of losses per task.
    metrics : Union[List[str], Dict[str, str]]
        List of metrics or dictionary of metrics per task.

    Returns
    -------
    model : Model
        The built model.
    """
    label_names = label_names if label_names is not None else [f'label_{i}' for i in range(n_tasks)]
    attention_dropouts = attention_dropouts if attention_dropouts is not None \
        else [0.1 for _ in range(n_attention_layers)]
    attention_key_dims = attention_key_dims if attention_key_dims is not None \
        else [8 for _ in range(n_attention_layers)]
    last_layers_units = last_layers_units if last_layers_units is not None else [1] * n_tasks
    last_layers_activations = last_layers_activations if last_layers_activations is not None else ['sigmoid'] * n_tasks
    losses = losses if losses is not None else ['binary_crossentropy'] * n_tasks
    metrics = metrics if metrics is not None else ['accuracy']
    assert len(label_names) == n_tasks == len(last_layers_units) == len(last_layers_activations)
    assert len(attention_dropouts) == n_attention_layers
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=input_dim, output_dim=embedding_output_dim)(input_layer)
    # Normalization layer
    normalization_layer = LayerNormalization()(embedding_layer)

    # Attention layers
    for i in range(n_attention_layers):
        attention_layer = MultiHeadAttention(num_heads=n_attention_heads, key_dim=attention_key_dims[i],
                                             dropout=attention_dropouts[i])(normalization_layer, normalization_layer)
        normalization_layer = LayerNormalization()(attention_layer)

    # Flatten layer
    flatten_layer = Flatten()(normalization_layer)

    # Output layers
    output_layers = []
    for i in range(n_tasks):
        output_layer = Dense(units=dense_units, activation=dense_activation)(flatten_layer)
        output_layer = Dropout(dense_dropout)(output_layer)
        output_layer = Dense(units=last_layers_units[i], activation=last_layers_activations[i],
                             name=label_names[i])(output_layer)
        output_layers.append(output_layer)

    # Model
    model = Model(inputs=input_layer, outputs=output_layers)

    # Compile model
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def keras_tabular_transformer_model(model_kwargs: dict = None,
                                    keras_kwargs: dict = None) -> KerasModel:
    """
    Build a transformer model using Keras.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the Keras model.

    Returns
    -------
    model : KerasModel
        The built model.
    """
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    keras_kwargs = keras_kwargs if keras_kwargs is not None else {}
    mode = keras_kwargs.get('mode', 'classification')
    epochs = keras_kwargs.get('epochs', 150)
    batch_size = keras_kwargs.get('batch_size', 10)
    verbose = keras_kwargs.get('verbose', 0)
    return KerasModel(model_builder=keras_tabular_transformer_model_builder, mode=mode,
                      epochs=epochs, batch_size=batch_size, verbose=verbose, **model_kwargs)


def keras_simple_rnn_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None,
                                   n_rnn_layers: int = 1, rnn_units: List[int] = None, rnn_dropouts: List[float] = None,
                                   dense_units: int = 64, dense_activation: str = 'relu', dense_dropout: float = 0.1,
                                   last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                                   optimizer: str = 'adam', losses: Union[List[str], Dict[str, str]] = None,
                                   metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    """
    Build a simple RNN model using Keras.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    n_tasks : int
        Number of tasks.
    label_names : List[str]
        List of label names.
    n_rnn_layers : int
        Number of RNN layers.
    rnn_units : List[int]
        List of units in the RNN layers.
    rnn_dropouts : List[float]
        List of dropout rates in the RNN layers.
    dense_units : int
        Number of units in the dense layer.
    dense_activation : str
        Activation function of the dense layer.
    dense_dropout : float
        Dropout rate in the dense layer.
    last_layers_units : List[int]
        List of units in the last layers.
    last_layers_activations : List[str]
        List of activation functions in the last layers.
    optimizer : str
        Optimizer.
    losses : Union[List[str], Dict[str, str]]
        Losses.
    metrics : Union[List[str], Dict[str, str]]
        Metrics.

    Returns
    -------
    model : Model
        The built model.
    """
    label_names = label_names if label_names is not None else [f'label_{i}' for i in range(n_tasks)]
    rnn_units = rnn_units if rnn_units is not None else [64 for _ in range(n_rnn_layers)]
    rnn_dropouts = rnn_dropouts if rnn_dropouts is not None else [0.1 for _ in range(n_rnn_layers)]
    last_layers_units = last_layers_units if last_layers_units is not None else [1] * n_tasks
    last_layers_activations = last_layers_activations if last_layers_activations is not None else ['sigmoid'] * n_tasks
    losses = losses if losses is not None else ['binary_crossentropy'] * n_tasks
    metrics = metrics if metrics is not None else ['accuracy']
    assert n_rnn_layers == len(rnn_units) == len(rnn_dropouts)
    assert len(label_names) == n_tasks == len(last_layers_units) == len(last_layers_activations)
    # Input Layer
    input_layer = Input(shape=input_dim)
    x = input_layer
    # RNN Layers
    for i in range(n_rnn_layers):
        x = SimpleRNN(units=rnn_units[i], return_sequences=True)(x)
        x = Dropout(rnn_dropouts[i])(x)

    # Flatten
    x = Flatten()(x)
    # Dense Output Layers
    outputs = []
    for i in range(n_tasks):
        task_layer = Dense(units=dense_units, activation=dense_activation)(x)
        task_layer = Dropout(dense_dropout)(task_layer)
        task_layer = Dense(units=last_layers_units[i], activation=last_layers_activations[i],
                           name=label_names[i])(task_layer)
        outputs.append(task_layer)

    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def keras_simple_rnn_model(model_kwargs: dict = None,
                           keras_kwargs: dict = None) -> KerasModel:
    """
    Build a simple RNN model using Keras.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the Keras model.

    Returns
    -------
    model : KerasModel
        The built model.
    """
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    keras_kwargs = keras_kwargs if keras_kwargs is not None else {}
    mode = keras_kwargs.get('mode', 'classification')
    epochs = keras_kwargs.get('epochs', 150)
    batch_size = keras_kwargs.get('batch_size', 10)
    verbose = keras_kwargs.get('verbose', 0)
    return KerasModel(model_builder=keras_simple_rnn_model_builder, mode=mode,
                      epochs=epochs, batch_size=batch_size, verbose=verbose, **model_kwargs)


def keras_rnn_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None,
                            n_lstm_layers: int = 1, lstm_units: List[int] = None, lstm_dropout: List[float] = None,
                            n_gru_layers: int = 0, gru_units: List[int] = None, gru_dropout: List[float] = None,
                            dense_units: int = 64, dense_dropout: float = 0.0, dense_activation: str = 'relu',
                            last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                            optimizer: str = 'adam', losses: Union[List[str], Dict[str, str]] = None,
                            metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    """
    Build a RNN model using Keras.

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    n_tasks : int
        Number of tasks.
    label_names : List[str]
        List of label names.
    n_lstm_layers : int
        Number of LSTM layers.
    lstm_units : int
        Number of units in the LSTM layers.
    lstm_dropout : float
        Dropout rate in the LSTM layers.
    n_gru_layers : int
        Number of GRU layers.
    gru_units : int
        Number of units in the GRU layers.
    gru_dropout : float
        Dropout rate in the GRU layers.
    dense_units : int
        Number of units in the dense layers.
    dense_dropout : float
        Dropout rate in the dense layers.
    dense_activation : str
        Activation function in the dense layers.
    last_layers_units : List[int]
        Number of units in the last layers.
    last_layers_activations : List[str]
        Activation functions in the last layers.
    optimizer : str
        Optimizer.
    losses : Union[List[str], Dict[str, str]]
        Loss functions.
    metrics : Union[List[str], Dict[str, str]]
        Metrics.

    Returns
    -------
    model : Model
        The built model.
    """
    label_names = label_names if label_names is not None else [f'label_{i}' for i in range(n_tasks)]
    lstm_units = lstm_units if lstm_units is not None else [64 for _ in range(n_lstm_layers)]
    lstm_dropout = lstm_dropout if lstm_dropout is not None else [0.1 for _ in range(n_lstm_layers)]
    gru_units = gru_units if gru_units is not None else [64 for _ in range(n_gru_layers)]
    gru_dropout = gru_dropout if gru_dropout is not None else [0.1 for _ in range(n_gru_layers)]
    last_layers_units = last_layers_units if last_layers_units is not None else [1] * n_tasks
    last_layers_activations = last_layers_activations if last_layers_activations is not None else ['sigmoid'] * n_tasks
    losses = losses if losses is not None else ['binary_crossentropy'] * n_tasks
    metrics = metrics if metrics is not None else ['accuracy']
    assert len(label_names) == len(last_layers_units) == len(last_layers_activations) == n_tasks
    assert len(lstm_units) == len(lstm_dropout) == n_lstm_layers
    assert len(gru_units) == len(gru_dropout) == n_gru_layers
    # Input Layer
    input_layer = Input(shape=input_dim)
    x = input_layer
    # LSTM Layers
    for i in range(n_lstm_layers):
        x = LSTM(units=lstm_units[i], dropout=lstm_dropout[i], return_sequences=True)(x)
    # GRU Layers
    for i in range(n_gru_layers):
        x = GRU(units=gru_units[i], dropout=gru_dropout[i], return_sequences=True)(x)

    # Flatten
    x = Flatten()(x)

    # Dense Layers
    outputs = []
    for i in range(n_tasks):
        task_layer = Dense(units=dense_units, activation=dense_activation)(x)
        task_layer = Dropout(rate=dense_dropout)(task_layer)
        task_layer = Dense(units=last_layers_units[i], activation=last_layers_activations[i],
                           name=label_names[i])(task_layer)
        outputs.append(task_layer)
    # Model
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def keras_rnn_model(model_kwargs: dict = None,
                    keras_kwargs: dict = None) -> KerasModel:
    """
    Build a RNN model using Keras.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the Keras model.

    Returns
    -------
    model : KerasModel
        The built model.
    """
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    keras_kwargs = keras_kwargs if keras_kwargs is not None else {}
    mode = keras_kwargs.get('mode', 'classification')
    epochs = keras_kwargs.get('epochs', 150)
    batch_size = keras_kwargs.get('batch_size', 10)
    verbose = keras_kwargs.get('verbose', 0)
    return KerasModel(model_builder=keras_rnn_model_builder, mode=mode,
                      epochs=epochs, batch_size=batch_size, verbose=verbose, **model_kwargs)


def keras_bidirectional_rnn_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None,
                                          n_lstm_layers: int = 1, lstm_units: List[int] = None,
                                          lstm_dropout: List[float] = None, n_gru_layers: int = 0,
                                          gru_units: List[int] = None, gru_dropout: List[float] = None,
                                          dense_units: int = 64, dense_dropout: float = 0.0,
                                          dense_activation: str = 'relu', last_layers_units: List[int] = None,
                                          last_layers_activations: List[str] = None,
                                          optimizer: str = 'adam', losses: Union[List[str], Dict[str, str]] = None,
                                          metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    """
    Build a bidirectional RNN model using Keras.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    n_tasks : int
        Number of tasks.
    label_names : List[str]
        Names of the labels.
    n_lstm_layers : int
        Number of LSTM layers.
    lstm_units : int
        Number of units in each LSTM layer.
    lstm_dropout : float
        Dropout rate in each LSTM layer.
    n_gru_layers : int
        Number of GRU layers.
    gru_units : int
        Number of units in each GRU layer.
    gru_dropout : float
        Dropout rate in each GRU layer.
    dense_units : int
        Number of units in the dense layer.
    dense_dropout : float
        Dropout rate in the dense layer.
    dense_activation : str
        Activation function in the dense layer.
    last_layers_units : List[int]
        Number of units in the last layers.
    last_layers_activations : List[str]
        Activation functions in the last layers.
    optimizer : str
        Optimizer.
    losses : Union[List[str], Dict[str, str]]
        Loss functions.
    metrics : Union[List[str], Dict[str, str]]
        Metrics.

    Returns
    -------
    model : Model
        The built model.
    """
    label_names = label_names if label_names is not None else [f'label_{i}' for i in range(n_tasks)]
    lstm_units = lstm_units if lstm_units is not None else [64 for _ in range(n_lstm_layers)]
    lstm_dropout = lstm_dropout if lstm_dropout is not None else [0.1 for _ in range(n_lstm_layers)]
    gru_units = gru_units if gru_units is not None else [64 for _ in range(n_gru_layers)]
    gru_dropout = gru_dropout if gru_dropout is not None else [0.1 for _ in range(n_gru_layers)]
    last_layers_units = last_layers_units if last_layers_units is not None else [1] * n_tasks
    last_layers_activations = last_layers_activations if last_layers_activations is not None else ['sigmoid'] * n_tasks
    losses = losses if losses is not None else ['binary_crossentropy'] * n_tasks
    metrics = metrics if metrics is not None else ['accuracy']
    assert len(label_names) == len(last_layers_units) == len(last_layers_activations) == n_tasks
    assert len(lstm_units) == len(lstm_dropout) == n_lstm_layers
    assert len(gru_units) == len(gru_dropout) == n_gru_layers
    # Input Layer
    input_layer = Input(shape=input_dim)
    x = input_layer
    # LSTM Layers
    for i in range(n_lstm_layers):
        if i == 0:
            x = Bidirectional(LSTM(lstm_units[i], return_sequences=True), input_shape=(input_dim))(x)
        else:
            x = Bidirectional(LSTM(lstm_units[i], return_sequences=True))(x)

        # Add GRU layers
    for j in range(n_gru_layers):
        if n_lstm_layers == 0 and j == 0:
            x = Bidirectional(GRU(gru_units[j], return_sequences=True), input_shape=(input_dim))(x)
        else:
            x = Bidirectional(GRU(gru_units[j], return_sequences=True))(x)

    # Flatten
    x = Flatten()(x)

    # Dense Layers
    outputs = []
    for i in range(n_tasks):
        task_layer = Dense(units=dense_units, activation=dense_activation)(x)
        task_layer = Dropout(rate=dense_dropout)(task_layer)
        task_layer = Dense(units=last_layers_units[i], activation=last_layers_activations[i],
                           name=label_names[i])(task_layer)
        outputs.append(task_layer)
    # Model
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model


def keras_bidirectional_rnn_model(model_kwargs: dict = None,
                                  keras_kwargs: dict = None) -> KerasModel:
    """
    Build a bidirectional RNN model using Keras.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the Keras model.

    Returns
    -------
    model : KerasModel
        The built model.
    """
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    keras_kwargs = keras_kwargs if keras_kwargs is not None else {}
    mode = keras_kwargs.get('mode', 'classification')
    epochs = keras_kwargs.get('epochs', 150)
    batch_size = keras_kwargs.get('batch_size', 10)
    verbose = keras_kwargs.get('verbose', 0)
    return KerasModel(model_builder=keras_bidirectional_rnn_model_builder, mode=mode,
                      epochs=epochs, batch_size=batch_size, verbose=verbose, **model_kwargs)
