from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, Conv1D, Flatten, Reshape
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop


# TODO: add more pre-defined models


def rf_model_builder(n_estimators=100, max_features='auto', class_weight=None):
    if class_weight is None:
        class_weight = {0: 1., 1: 1.}
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                      class_weight=class_weight)
    return rf_model


def svm_model_builder(C=1, gamma='auto', kernel='rfb'):
    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
    return svm_model


def create_dense_model(input_dim=1024,
                       n_hidden_layers=1,
                       layers_units=None,
                       dropouts=None,
                       activations=None,
                       batch_normalization=None,
                       l1_l2=None,
                       loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=None):
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
                        kernel_regularizer=regularizers.l1_l2(l1=l1_l2[i + 1][0], l2=l1_l2[i + 1][1])))
        if batch_normalization[i + 1]:
            model.add(BatchNormalization())
        model.add(Dropout(dropouts[i + 1]))
    model.add(Dense(layers_units[-1], activation=activations[-1]))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def make_cnn_model(input_dim=1024,
                   g_noise=0.05,
                   DENSE=128,
                   DROPOUT=0.5,
                   C1_K=8,
                   C1_S=32,
                   C2_K=16,
                   C2_S=32,
                   activation='relu',
                   loss='binary_crossentropy',
                   optimizer='adadelta',
                   learning_rate=0.01,
                   metrics='accuracy'):
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
