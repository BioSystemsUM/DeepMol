from typing import List, Union

from tensorflow.keras import Sequential, regularizers, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, Reshape, Conv1D, Flatten, \
    InputLayer, Embedding, MultiHeadAttention, LayerNormalization, Input, SimpleRNN, LSTM, GRU, Bidirectional

from deepmol.models import KerasModel


def baseline_dense_model_builder(input_dim: int = 1024,
                                 n_hidden_layers: int = 1,
                                 layers_units: List[int] = None,
                                 dropouts: List[float] = None,
                                 activations: List[str] = None,
                                 batch_normalization: List[bool] = None,
                                 l1_l2: List[float] = None,
                                 loss: str = 'binary_crossentropy',
                                 optimizer: str = 'adam',
                                 metrics: List[str] = None):
    """
    Builds a dense neural network model.

    Parameters
    ----------
    input_dim : int
        Number of features.
    n_hidden_layers : int
        Number of hidden layers.
    layers_units : List[int]
        Number of units in each hidden layer.
    dropouts : List[float]
        Dropout rate in each hidden layer.
    activations : List[str]
        Activation function in each hidden layer.
    batch_normalization : List[bool]
        Whether to use batch normalization in each hidden layer.
    l1_l2 : List[float]
        L1 and L2 regularization in each hidden layer.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metrics : List[str]
        Metrics to be evaluated by the model during training and testing.

    Returns
    -------
    model : Sequential
        Dense neural network model.
    """
    metrics = ['accuracy'] if metrics is None else metrics
    l1_l2 = [(0, 0)] if l1_l2 is None else l1_l2
    batch_normalization = [True, True] if batch_normalization is None else batch_normalization
    activations = ['relu', 'relu', 'sigmoid'] if activations is None else activations
    layers_units = [12, 8, 1] if layers_units is None else layers_units
    dropouts = [0.5, 0.5] if dropouts is None else dropouts
    model = Sequential()
    model.add(Dense(layers_units[0], input_dim=input_dim, activation=activations[0]))
    if batch_normalization[0]:
        model.add(BatchNormalization())
    model.add(Dropout(dropouts[0]))
    for i in range(n_hidden_layers):
        model.add(Dense(layers_units[i + 1], activation=activations[i + 1],
                        kernel_regularizer=regularizers.l1_l2(l1=l1_l2[i][0], l2=l1_l2[i][1])))
        if batch_normalization[i + 1]:
            model.add(BatchNormalization())
        model.add(Dropout(dropouts[i + 1]))
    model.add(Dense(layers_units[-1], activation=activations[-1]))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def keras_dense_model(model_dir: str, model_kwargs: dict = None, keras_kwargs: dict = None) -> KerasModel:
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    model_path = model_dir
    loss = 'binary_crossentropy' if 'loss' not in keras_kwargs else keras_kwargs['loss']
    optimizer = 'adam' if 'optimizer' not in keras_kwargs else keras_kwargs['optimizer']
    learning_rate = 0.001 if 'learning_rate' not in keras_kwargs else keras_kwargs['learning_rate']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=baseline_dense_model_builder, mode=mode, model_path=model_path, loss=loss,
                      optimizer=optimizer, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                      verbose=verbose, **model_kwargs)


def make_cnn_model_builder(input_dim: int = 1024,
                           g_noise: float = 0.05,
                           n_conv_layers: int = 2,
                           filters: List[int] = [8, 16],
                           kernel_sizes: List[int] = [32, 32],
                           strides: List[int] = [1, 1],
                           activations: List[str] = 'relu',
                           padding: str = 'same',
                           dense_units: int = 128,
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
    model.add(Dense(dense_units, activation=activations))
    model.add(Dense(last_layer_units, activation=last_layer_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def keras_cnn_model(model_dir: str, model_kwargs: dict = None, keras_kwargs: dict = None) -> KerasModel:
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    model_path = model_dir
    loss = 'binary_crossentropy' if 'loss' not in keras_kwargs else keras_kwargs['loss']
    optimizer = 'adam' if 'optimizer' not in keras_kwargs else keras_kwargs['optimizer']
    learning_rate = 0.001 if 'learning_rate' not in keras_kwargs else keras_kwargs['learning_rate']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=baseline_dense_model_builder, mode=mode, model_path=model_path, loss=loss,
                      optimizer=optimizer, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                      verbose=verbose, **model_kwargs)


def make_tabular_transformer_model_builder(input_dim: int, embedding_output_dim: int = 32, n_attention_layers: int = 2,
                                           n_heads: int = 4, dropout_attention_l: float = 0.1, dense_units: int = 64,
                                           last_layer_units: int = 1, last_layer_activation: str = 'sigmoid',
                                           loss: str = 'binary_crossentropy', optimizer: str = 'adam',
                                           metric: str = 'accuracy'):
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


def keras_tabular_transformer_model(model_dir: str, model_kwargs: dict = None, keras_kwargs: dict = None) -> KerasModel:
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    model_path = model_dir
    loss = 'binary_crossentropy' if 'loss' not in keras_kwargs else keras_kwargs['loss']
    optimizer = 'adam' if 'optimizer' not in keras_kwargs else keras_kwargs['optimizer']
    learning_rate = 0.001 if 'learning_rate' not in keras_kwargs else keras_kwargs['learning_rate']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_tabular_transformer_model_builder, mode=mode, model_path=model_path, loss=loss,
                      optimizer=optimizer, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                      verbose=verbose, **model_kwargs)


def make_simple_rnn_model_builder(input_dim: tuple, n_rnn_layers: int = 1, rnn_units: int = 64,
                                  dropout_rnn: float = 0.1, dense_units: int = 64, last_layer_units: int = 1,
                                  last_layer_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                                  optimizer: str = 'adam', metric: str = 'accuracy'):
    inputs = Input(shape=input_dim)
    x = inputs
    for _ in range(n_rnn_layers):
        x = SimpleRNN(rnn_units, return_sequences=True)(x)
        x = Dropout(dropout_rnn)(x)
    x = Flatten()(x)
    x = Dense(units=dense_units, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(units=last_layer_units, activation=last_layer_activation)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    return model


def keras_simple_rnn_model(model_dir: str, model_kwargs: dict = None, keras_kwargs: dict = None) -> KerasModel:
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    model_path = model_dir
    loss = 'binary_crossentropy' if 'loss' not in keras_kwargs else keras_kwargs['loss']
    optimizer = 'adam' if 'optimizer' not in keras_kwargs else keras_kwargs['optimizer']
    learning_rate = 0.001 if 'learning_rate' not in keras_kwargs else keras_kwargs['learning_rate']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_simple_rnn_model_builder, mode=mode, model_path=model_path, loss=loss,
                      optimizer=optimizer, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                      verbose=verbose, **model_kwargs)


def make_rnn_model_builder(num_lstm_layers: int = 1, lstm_units: List[int] = [128], num_gru_layers: int = 1,
                           gru_units: List[int] = [128], dense_units: int = 64, last_layer_units: int = 1,
                           last_layer_activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                           optimizer: str = 'adam', metric: str = 'accuracy'):
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


def keras_rnn_model(model_dir: str, model_kwargs: dict = None, keras_kwargs: dict = None) -> KerasModel:
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    model_path = model_dir
    loss = 'binary_crossentropy' if 'loss' not in keras_kwargs else keras_kwargs['loss']
    optimizer = 'adam' if 'optimizer' not in keras_kwargs else keras_kwargs['optimizer']
    learning_rate = 0.001 if 'learning_rate' not in keras_kwargs else keras_kwargs['learning_rate']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_rnn_model_builder, mode=mode, model_path=model_path, loss=loss,
                      optimizer=optimizer, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                      verbose=verbose, **model_kwargs)


def make_bidirectional_rnn_model_builder(input_dim: tuple, num_lstm_layers: int = 1, lstm_units: List[int] = [128],
                                         num_gru_layers: int = 1, gru_units: List[int] = [128], dense_units: int = 64,
                                         last_layer_units: int = 1, last_layer_activation: str = 'sigmoid',
                                         loss: str = 'binary_crossentropy', optimizer: str = 'adam',
                                         metric: str = 'accuracy'):
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


def keras_bidirectional_rnn_model(model_dir: str, model_kwargs: dict = None, keras_kwargs: dict = None) -> KerasModel:
    mode = 'classification' if 'mode' not in keras_kwargs else keras_kwargs['mode']
    model_path = model_dir
    loss = 'binary_crossentropy' if 'loss' not in keras_kwargs else keras_kwargs['loss']
    optimizer = 'adam' if 'optimizer' not in keras_kwargs else keras_kwargs['optimizer']
    learning_rate = 0.001 if 'learning_rate' not in keras_kwargs else keras_kwargs['learning_rate']
    epochs = 150 if 'epochs' not in keras_kwargs else keras_kwargs['epochs']
    batch_size = 10 if 'batch_size' not in keras_kwargs else keras_kwargs['batch_size']
    verbose = 0 if 'verbose' not in keras_kwargs else keras_kwargs['verbose']
    model_kwargs = {} if model_kwargs is None else model_kwargs
    return KerasModel(model_builder=make_bidirectional_rnn_model_builder, mode=mode, model_path=model_path, loss=loss,
                      optimizer=optimizer, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                      verbose=verbose, **model_kwargs)
