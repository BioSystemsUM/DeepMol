from ast import literal_eval

from deepchem.models import GraphConvModel, TextCNNModel, WeaveModel, MPNNModel
from deepchem.models import torch_models
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from src.models.DeepChemModels import DeepChemModel


def graphconv_builder(graph_conv_layers, dense_layer_size, dropout, learning_rate, task_type, batch_size=256,
                      epochs=100):
    graph = GraphConvModel(n_tasks=1, graph_conv_layers=graph_conv_layers, dense_layer_size=dense_layer_size,
                           dropout=dropout, batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
    return DeepChemModel(graph, epochs=epochs, use_weights=False, model_dir=None)


# optimizer = Adam by default in DeepChem and loss=L2Loss() by default for regression (it's the same as MSE loss)


def textcnn_builder(char_dict, seq_length, n_embedding, kernel_sizes, num_filters, dropout, learning_rate, task_type,
                    batch_size=256, epochs=100):
    textcnn = TextCNNModel(n_tasks=1, char_dict=char_dict, seq_length=seq_length, n_embedding=n_embedding,
                           kernel_sizes=kernel_sizes, num_filters=num_filters, dropout=dropout,
                           batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
    return DeepChemModel(textcnn, epochs=epochs, use_weights=False, model_dir=None)


def weave_builder(n_hidden, n_graph_feat, n_weave, fully_connected_layer_sizes, dropouts,
                  learning_rate, task_type, batch_size=256, epochs=100):
    weave = WeaveModel(n_tasks=1, n_hidden=n_hidden, n_graph_feat=n_graph_feat, n_weave=n_weave,
                       fully_connected_layer_sizes=fully_connected_layer_sizes,
                       dropouts=dropouts, batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
    return DeepChemModel(weave, epochs=epochs, use_weights=False, model_dir=None)


def mpnn_builder(n_hidden, T, M, learning_rate, task_type, batch_size=256,
                 epochs=100):
    mpnn = MPNNModel(n_tasks=1, n_atom_feat=75, n_pair_feat=14, n_hidden=n_hidden, T=T, M=M, dropout=0,
                     learning_rate=learning_rate, batch_size=batch_size, mode=task_type)
    # the dropout arg isn't actually used in the DeepChem code for MPNNModel!
    return DeepChemModel(mpnn, epochs=epochs, use_weights=False, model_dir=None)


def gcn_builder(graph_conv_layers, dropout, predictor_hidden_feats, predictor_dropout, learning_rate, task_type,
                batch_size=256, epochs=100):
    gcn = torch_models.GCNModel(n_tasks=1, graph_conv_layers=graph_conv_layers, activation=None,
                                # if activation=None, DGLLife defaults to ReLU
                                residual=True, batchnorm=False, predictor_hidden_feats=predictor_hidden_feats,
                                dropout=dropout, predictor_dropout=predictor_dropout, learning_rate=learning_rate,
                                batch_size=batch_size, mode=task_type)
    return DeepChemModel(gcn, epochs=epochs, use_weights=False, model_dir=None)


def gat_builder(graph_attention_layers, n_attention_heads, dropout, predictor_hidden_feats,
                predictor_dropout, learning_rate, task_type, batch_size=256, epochs=100):
    gat = torch_models.GATModel(n_tasks=1, graph_attention_layers=graph_attention_layers,
                                n_attention_heads=n_attention_heads, dropout=dropout,
                                predictor_hidden_feats=predictor_hidden_feats, predictor_dropout=predictor_dropout,
                                learning_rate=learning_rate, batch_size=batch_size, mode=task_type)
    return DeepChemModel(gat, epochs=epochs, use_weights=False, model_dir=None)


def attentivefp_builder(num_layers, num_timesteps, graph_feat_size, dropout, learning_rate, task_type, batch_size=256,
                        epochs=100):
    attentivefp = torch_models.AttentiveFPModel(n_tasks=1, num_layers=num_layers, num_timesteps=num_timesteps,
                                                graph_feat_size=graph_feat_size, dropout=dropout,
                                                learning_rate=learning_rate, batch_size=batch_size, mode=task_type)
    return DeepChemModel(attentivefp, epoch=epochs, use_weights=False, model_dir=None)


def torchmpnn_builder(node_out_feats, edge_hidden_feats, num_step_message_passing, num_step_set2set, num_layer_set2set,
                      learning_rate, task_type, batch_size=256, epochs=100):
    mpnn = torch_models.MPNNModel(n_tasks=1, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats,
                                  num_step_message_passing=num_step_message_passing, num_step_set2set=num_step_set2set,
                                  num_layer_set2set=num_layer_set2set, learning_rate=learning_rate,
                                  batch_size=batch_size, mode=task_type)
    return DeepChemModel(mpnn, epoch=epochs, use_weights=False, model_dir=None)


def dense_builder(input_dim=None, task_type=None, hlayers_sizes='[10]', initializer='he_normal',
                  l1=0, l2=0, hidden_dropout=0, batchnorm=True, learning_rate=0.001):
    hlayers_sizes = literal_eval(hlayers_sizes)
    # hlayers_sizes was passed as a str because Pipeline throws RuntimeError when cloning the model if parameters are lists

    model = Sequential()
    model.add(Input(shape=input_dim))

    for i in range(len(hlayers_sizes)):
        model.add(Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
                        kernel_regularizer=l1_l2(l1=l1, l2=l2)))
        if batchnorm:
            model.add(BatchNormalization())

        model.add(Activation('relu'))

        if hidden_dropout > 0:
            model.add(Dropout(rate=hidden_dropout))

    # Add output layer
    if task_type == 'regression':
        model.add(Dense(1, activation='linear', kernel_initializer=initializer))
    elif task_type == 'classification':
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

    # Define optimizer
    opt = Adam(lr=learning_rate)

    # Compile model
    if task_type == 'regression':
        model.compile(loss='mean_squared_error', optimizer=opt)
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


BUILDERS = {'GraphConv': graphconv_builder,
            'ECFP4': dense_builder,
            'ECFP6': dense_builder,
            'Mol2vec': dense_builder,
            'TextCNN': textcnn_builder,
            'Weave': weave_builder,
            'MACCS': dense_builder,
            'RDKitFP': dense_builder,
            'MPNN': mpnn_builder,
            'GCN': gcn_builder,
            'GAT': gat_builder,
            'AttentiveFP': attentivefp_builder,
            'TorchMPNN': torchmpnn_builder,
            'AtomPair': dense_builder,
            'LayeredFP': dense_builder}
