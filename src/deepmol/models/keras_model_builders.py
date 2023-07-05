from typing import List, Union, Tuple, Dict

from tensorflow.keras import Sequential, regularizers, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, Reshape, Conv1D, Flatten, \
    InputLayer, Embedding, MultiHeadAttention, LayerNormalization, Input, SimpleRNN, LSTM, GRU, Bidirectional

from deepmol.models import KerasModel


def keras_fcnn_model_builder(input_dim: int, n_tasks: int = 1, label_names: List[str] = None, n_hidden_layers: int = 1,
                             hidden_units: List[int] = None, hidden_activations: List[str] = None,
                             hidden_regularizers: List[Tuple[float, float]] = None,
                             hidden_dropouts: List[float] = None, batch_normalization: List[bool] = None,
                             last_layers_units: List[int] = None, last_layers_activations: List[str] = None,
                             optimizer: str = 'adam', losses: Union[List[str], Dict[str, str]] = None,
                             metrics: Union[List[str], Dict[str, str]] = None) -> Model:
    label_names = label_names if label_names is not None else ['y']
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


def keras_fcnn_model(model_dir: str = 'keras_model/', model_kwargs: dict = None,
                     keras_kwargs: dict = None) -> KerasModel:
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=keras_fcnn_model_builder, mode=mode, model_dir=model_dir, epochs=epochs,
                      batch_size=batch_size, verbose=verbose, **model_kwargs)


def baseline_1D_cnn_model_builder(input_dim: int = 1024,
                                  g_noise: float = 0.05,
                                  n_conv_layers: int = 2,
                                  filters: List[int] = [8, 16],
                                  kernel_sizes: List[int] = [32, 32],
                                  strides: List[int] = [1, 1],
                                  activations: List[str] = ['relu', 'relu'],
                                  padding: str = 'same',
                                  dense_units: int = 128,
                                  dense_activation: str = 'relu',
                                  dropout: float = 0.5,
                                  last_layer_units: int = 1,
                                  last_layer_activation: str = 'sigmoid',
                                  loss: str = 'binary_crossentropy',
                                  optimizer: str = 'adadelta',
                                  metrics: Union[str, List[str]] = 'accuracy'):
    """
    Builds a 1D convolutional neural network model.

    Parameters
    ----------
    input_dim : int
        Number of features.
    g_noise : float
        Gaussian noise standard deviation.
    n_conv_layers : int
        Number of convolutional layers.
    filters : List[int]
        Number of filters in each convolutional layer.
    kernel_sizes : List[int]
        Kernel size in each convolutional layer.
    strides : List[int]
        Stride in each convolutional layer.
    activations : List[str]
        Activation function in each convolutional layer.
    padding : str
        One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with
        zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as
        the input. "causal" results in causal (dilated) convolutions.
    dense_units : int
        Number of units in the dense layer.
    dense_activation : str
        Activation function in the dense layer.
    dropout : float
        Dropout rate in the dense layer.
    last_layer_units : int
        Number of units in the last layer.
    last_layer_activation : str
        Activation function in the last layer.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metrics : List[str]
        Metrics to be evaluated by the model during training and testing.
    """
    assert len(filters) == len(kernel_sizes) == len(activations) == n_conv_layers, \
        'Number of filters, kernel sizes, activations and number of convolutional layers must be the same.'
    model = Sequential()
    # Adding a bit of GaussianNoise also works as regularization
    model.add(GaussianNoise(g_noise, input_shape=(input_dim,)))
    # First two is number of filter + kernel size
    model.add(Reshape((input_dim, 1)))
    for i in range(n_conv_layers):
        model.add(Conv1D(filters[i], kernel_sizes[i], activation=activations[i], strides=strides[i], padding=padding))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(dense_units, activation=dense_activation))
    model.add(Dense(last_layer_units, activation=last_layer_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def keras_1D_cnn_model(model_dir: str = 'keras_model/', model_kwargs: dict = None,
                       keras_kwargs: dict = None) -> KerasModel:
    """
    Builds a convolutional neural network model using DeepMol's KerasModel wrapper.

    Parameters
    ----------
    model_dir : str
        Path to save the model.
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the KerasModel wrapper.

    Returns
    -------
    model : KerasModel
        Convolutional neural network model.
    """
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=baseline_1D_cnn_model_builder, mode=mode, model_dir=model_dir, epochs=epochs,
                      batch_size=batch_size, verbose=verbose, **model_kwargs)


def make_tabular_transformer_model_builder(input_dim: int, embedding_output_dim: int = 32, n_attention_layers: int = 2,
                                           n_heads: int = 4, dropout_attention_l: float = 0.1, dense_units: int = 64,
                                           last_layer_units: int = 1, last_layer_activation: str = 'sigmoid',
                                           loss: str = 'binary_crossentropy', optimizer: str = 'adam',
                                           metric: str = 'accuracy'):
    """
    Builds a tabular transformer model.

    Parameters
    ----------
    input_dim : int
        Number of features.
    embedding_output_dim : int
        Dimension of the embedding output.
    n_attention_layers : int
        Number of attention layers.
    n_heads : int
        Number of attention heads.
    dropout_attention_l : float
        Dropout rate in the attention layers.
    dense_units : int
        Number of units in the dense layer.
    last_layer_units : int
        Number of units in the last layer.
    last_layer_activation : str
        Activation function in the last layer.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metric : str
        Metric to be evaluated by the model during training and testing.

    Returns
    -------
    model : Model
        Tabular transformer model.
    """
    inputs = Input(shape=(input_dim,))

    # Positional encoding
    x = Embedding(input_dim, embedding_output_dim)(inputs)
    x = LayerNormalization()(x)

    # Transformer Encoder layers
    for _ in range(n_attention_layers):
        x = MultiHeadAttention(num_heads=n_heads, key_dim=8, dropout=dropout_attention_l)(x, x)
        x = LayerNormalization()(x)

    x = Flatten()(x)
    x = Dense(units=dense_units, activation="relu")(x)
    x = Dropout(0.1)(x)

    # Output layer
    outputs = Dense(units=last_layer_units, activation=last_layer_activation)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model


def keras_tabular_transformer_model(model_dir: str = 'keras_model/', model_kwargs: dict = None,
                                    keras_kwargs: dict = None) -> KerasModel:
    """
    Builds a tabular transformer model using DeepMol's KerasModel wrapper.

    Parameters
    ----------
    model_dir : str
        Path to save the model.
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the KerasModel wrapper.

    Returns
    -------
    model : KerasModel
        Tabular transformer model.
    """
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_tabular_transformer_model_builder, mode=mode, model_dir=model_dir,
                      epochs=epochs, batch_size=batch_size, verbose=verbose, **model_kwargs)


def make_simple_rnn_model_builder(input_dim: tuple, n_rnn_layers: int = 1, rnn_units: int = 64,
                                  dropout_rnn: float = 0.1, dense_units: int = 64, dense_dropout: float = 0.1,
                                  last_layer_units: int = 1, last_layer_activation: str = 'sigmoid',
                                  loss: str = 'binary_crossentropy', optimizer: str = 'adam', metric: str = 'accuracy'):
    """
    Builds a simple recurrent neural network model.

    Parameters
    ----------
    input_dim : tuple
        Input shape.
    n_rnn_layers : int
        Number of recurrent layers.
    rnn_units : int
        Number of units in the recurrent layer.
    dropout_rnn : float
        Dropout rate in the recurrent layers.
    dense_units : int
        Number of units in the dense layer.
    dense_dropout : float
        Dropout rate in the dense layer.
    last_layer_units : int
        Number of units in the last layer.
    last_layer_activation : str
        Activation function in the last layer.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metric : str
        Metric to be evaluated by the model during training and testing.

    Returns
    -------
    model : Model
        Simple recurrent neural network model.
    """
    inputs = Input(shape=input_dim)
    x = inputs
    for _ in range(n_rnn_layers):
        x = SimpleRNN(rnn_units, return_sequences=True)(x)
        x = Dropout(dropout_rnn)(x)
    x = Flatten()(x)
    x = Dense(units=dense_units, activation="relu")(x)
    x = Dropout(dense_dropout)(x)
    outputs = Dense(units=last_layer_units, activation=last_layer_activation)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model


def keras_simple_rnn_model(model_dir: str = 'keras_model/', model_kwargs: dict = None,
                           keras_kwargs: dict = None) -> KerasModel:
    """
    Builds a simple recurrent neural network model using DeepMol's KerasModel wrapper.

    Parameters
    ----------
    model_dir : str
        Path to save the model.
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the KerasModel wrapper.

    Returns
    -------
    model : KerasModel
        Simple recurrent neural network model.
    """
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_simple_rnn_model_builder, mode=mode, model_dir=model_dir, epochs=epochs,
                      batch_size=batch_size, verbose=verbose, **model_kwargs)


def make_rnn_model_builder(num_lstm_layers: int = 1, lstm_units: List[int] = [128], num_gru_layers: int = 1,
                           gru_units: List[int] = [128], dense_units: int = 64, last_layer_units: int = 1,
                           last_layer_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                           optimizer: str = 'adam', metric: str = 'accuracy'):
    """
    Builds a recurrent neural network model.

    Parameters
    ----------
    num_lstm_layers : int
        Number of LSTM layers.
    lstm_units : List[int]
        Number of units in each LSTM layer.
    num_gru_layers : int
        Number of GRU layers.
    gru_units : List[int]
        Number of units in each GRU layer.
    dense_units : int
        Number of units in the dense layer.
    last_layer_units : int
        Number of units in the last layer.
    last_layer_activation : str
        Activation function in the last layer.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metric : str
        Metric to be evaluated by the model during training and testing.

    Returns
    -------
    model : Model
        Recurrent neural network model.
    """
    assert len(lstm_units) == num_lstm_layers, "lstm_units must be a list of length num_lstm_layers"
    assert len(gru_units) == num_gru_layers, "gru_units must be a list of length num_gru_layers"
    model = Sequential()
    # Add LSTM layers
    for i in range(num_lstm_layers):
        model.add(LSTM(lstm_units[i], return_sequences=True))
    # Add GRU layers
    for j in range(num_gru_layers):
        model.add(GRU(gru_units[j], return_sequences=True))

    model.add(Flatten())

    # Add Dense layer
    model.add(Dense(units=dense_units, activation="relu"))
    model.add(Dense(last_layer_units, activation=last_layer_activation))

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model


def keras_rnn_model(model_dir: str = 'keras_model/', model_kwargs: dict = None,
                    keras_kwargs: dict = None) -> KerasModel:
    """
    Builds a recurrent neural network model using DeepMol's KerasModel wrapper.

    Parameters
    ----------
    model_dir : str
        Path to save the model.
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the KerasModel wrapper.

    Returns
    -------
    model : KerasModel
        Recurrent neural network model.
    """
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_rnn_model_builder, mode=mode, model_dir=model_dir, epochs=epochs,
                      batch_size=batch_size, verbose=verbose, **model_kwargs)


def make_bidirectional_rnn_model_builder(input_dim: tuple, num_lstm_layers: int = 1, lstm_units: List[int] = [128],
                                         num_gru_layers: int = 1, gru_units: List[int] = [128], dense_units: int = 64,
                                         last_layer_units: int = 1, last_layer_activation: str = 'sigmoid',
                                         loss: str = 'binary_crossentropy', optimizer: str = 'adam',
                                         metric: str = 'accuracy'):
    """
    Builds a bidirectional recurrent neural network model.

    Parameters
    ----------
    input_dim : tuple
        Input shape.
    num_lstm_layers : int
        Number of LSTM layers.
    lstm_units : List[int]
        Number of units in each LSTM layer.
    num_gru_layers : int
        Number of GRU layers.
    gru_units : List[int]
        Number of units in each GRU layer.
    dense_units : int
        Number of units in the dense layer.
    last_layer_units : int
        Number of units in the last layer.
    last_layer_activation : str
        Activation function in the last layer.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metric : str
        Metric to be evaluated by the model during training and testing.

    Returns
    -------
    model : Model
        Bidirectional recurrent neural network model.
    """
    model = Sequential()

    # Add LSTM layers
    for i in range(num_lstm_layers):
        if i == 0:
            model.add(Bidirectional(LSTM(lstm_units[i], return_sequences=True), input_shape=(input_dim)))
        else:
            model.add(Bidirectional(LSTM(lstm_units[i], return_sequences=True)))

    # Add GRU layers
    for j in range(num_gru_layers):
        if num_lstm_layers == 0 and j == 0:
            model.add(Bidirectional(GRU(gru_units[j], return_sequences=True), input_shape=(input_dim)))
        else:
            model.add(Bidirectional(GRU(gru_units[j], return_sequences=True)))

    model.add(Flatten())

    # Add Dense layer
    model.add(Dense(units=dense_units, activation="relu"))
    model.add(Dense(last_layer_units, activation=last_layer_activation))

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    return model


def keras_bidirectional_rnn_model(model_dir: str = 'keras_model/', model_kwargs: dict = None,
                                  keras_kwargs: dict = None) -> KerasModel:
    """
    Builds a bidirectional recurrent neural network model using DeepMol's KerasModel wrapper.

    Parameters
    ----------
    model_dir : str
        Path to save the model.
    model_kwargs : dict
        Keyword arguments for the model builder.
    keras_kwargs : dict
        Keyword arguments for the KerasModel wrapper.

    Returns
    -------
    model : KerasModel
        Bidirectional recurrent neural network model.
    """
    keras_kwargs = {} if keras_kwargs is None else keras_kwargs
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_bidirectional_rnn_model_builder, mode=mode, model_dir=model_dir,
                      epochs=epochs, batch_size=batch_size, verbose=verbose, **model_kwargs)
