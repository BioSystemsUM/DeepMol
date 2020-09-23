
from deepchem.models import GraphConvModel


class Models():

    def __init__(self):
        pass

    def x(self):
        pass


class ScikitLearnModels():

    def __init__(self):
        pass

    def x(self):
        pass

class DeepChemModels():

    def __init__(self):
        pass

    def GraphConvolutionModel(self, datasets, metric, transformers):

        train_dataset, valid_dataset, test_dataset = datasets

        model = GraphConvModel(n_tasks=1, graph_conv_layers=[64,64], dense_layer_size=128, dropout=0.0,
                               mode='classification', number_atom_features=75, n_classes=2, batch_size=100,
                               batch_normalize=True, uncertainty=False)

        model.fit(train_dataset, nb_epoch=20)

        #TODO: implement evaluation part in a independent class/functions
        print("Evaluating model")
        train_scores = model.evaluate(train_dataset, [metric], transformers)
        valid_scores = model.evaluate(valid_dataset, [metric], transformers)
        test_scores = model.evaluate(test_dataset, [metric], transformers)

        print("Train scores")
        print(train_scores)

        print("Validation scores")
        print(valid_scores)

        print("Test scores")
        print(test_scores)

