from typing import Union, List

from tensorflow.keras.optimizers import Adadelta, RMSprop, Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras import Sequential, regularizers, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv1D, Flatten, Input, BatchNormalization, GaussianNoise


# TODO: add more pre-defined models


def rf_model_builder(n_estimators: int = 100,
                     max_features: Union[int, float, str] = 'auto',
                     class_weight: dict = None):
    """
    Builds a random forest model.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_features : Union[int, float, str]
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at
        each split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
    class_weight : dict
        Weights associated with classes in the form `{class_label: weight}`.
        If not given, all classes are supposed to have weight one.

    Returns
    -------
    rf_model: RandomForestClassifier
        Random forest model.
    """
    if class_weight is None:
        class_weight = {0: 1., 1: 1.}
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                      class_weight=class_weight)
    return rf_model


def svm_model_builder(C: float = 1.0, gamma: Union[str, float] = 'auto', kernel: Union[str, callable] = 'rfb'):
    """
    Builds a support vector machine model.

    Parameters
    ----------
    C : float
        Penalty parameter C of the error term.
    gamma : Union[str, float]
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - if 'scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma;
            - if 'auto', uses 1 / n_features.
    kernel : str
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.

    Returns
    -------
    svm_model: SVC
        Support vector machine model.
    """
    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
    return svm_model


def create_dense_model(input_dim: int = 1024,
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
    # assert n_hidden_layers+2 = len(layers_units) == len(activations) == len(dropouts)+1 == len(
    # batch_normalization)+1 len(l1_l2)+2 create model
    if metrics is None:
        metrics = ['accuracy']
    if l1_l2 is None:
        l1_l2 = [(0, 0)]
    if batch_normalization is None:
        batch_normalization = [True, True]
    if activations is None:
        activations = ['relu', 'relu', 'sigmoid']
    if layers_units is None:
        layers_units = [12, 8, 1]
    if dropouts is None:
        dropouts = [0.5, 0.5]
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


def make_cnn_model(input_dim: int = 1024,
                   g_noise: float = 0.05,
                   DENSE: int = 128,
                   DROPOUT: float = 0.5,
                   C1_K: int = 8,
                   C1_S: int = 32,
                   C2_K: int = 16,
                   C2_S: int = 32,
                   activation: str = 'relu',
                   loss: str = 'binary_crossentropy',
                   optimizer: str = 'adadelta',
                   learning_rate: float = 0.01,
                   metrics: Union[str, List[str]] = 'accuracy'):
    """
    Builds a 1D convolutional neural network model.

    Parameters
    ----------
    input_dim : int
        Number of features.
    g_noise : float
        Gaussian noise.
    DENSE : int
        Number of units in the dense layer.
    DROPOUT : float
        Dropout rate.
    C1_K : int
        The dimensionality of the output space (i.e. the number of output filters in the convolution) of the first
        convolutional layer.
    C1_S : int
        Kernel size specifying the length of the 1D convolution window of the first convolutional layer.
    C2_K : int
        The dimensionality of the output space (i.e. the number of output filters in the convolution) of the second
        convolutional layer.
    C2_S : int
        Kernel size specifying the length of the 1D convolution window of the second convolutional layer.
    activation : str
        Activation function of the Conv1D and Dense layers.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    learning_rate : float
        Learning rate.
    metrics : Union[str, List[str]]
        Metrics to be evaluated by the model during training and testing.
    """
    model = Sequential()
    # Adding a bit of GaussianNoise also works as regularization
    model.add(GaussianNoise(g_noise, input_shape=(input_dim,)))
    # First two is number of filter + kernel size
    model.add(Reshape((input_dim, 1)))
    model.add(Conv1D(C1_K, C1_S, activation=activation, padding="same"))
    model.add(Conv1D(C2_K, C2_S, padding="same", activation=activation))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer == 'adadelta':
        opt = Adadelta(lr=learning_rate)
    elif optimizer == 'adam':
        opt = Adam(lr=learning_rate)
    elif optimizer == 'rsmprop':
        opt = RMSprop(lr=learning_rate)
    else:
        opt = optimizer

    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model


def basic_multitask_dnn(input_shape, task_names, losses, metrics):
    # Define the inputs
    inputs = Input(shape=input_shape)

    # Define the shared layers
    x = Dense(64, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)

    output_layers = []
    for i in range(len(task_names)):
        task_layer = Dense(16, activation="relu")(x)
        # Define the outputs for each task
        task_output = Dense(1, activation="sigmoid", name=f"{task_names[i]}")(task_layer)
        output_layers.append(task_output)

    # Define the model that outputs the predictions for each task
    model = Model(inputs=inputs, outputs=output_layers)
    losses = {task_names[i]: losses[i] for i in range(len(task_names))}
    metrics = {task_names[i]: metrics[i] for i in range(len(task_names))}
    # Compile the model with different loss functions and metrics for each task
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics)
    return model
