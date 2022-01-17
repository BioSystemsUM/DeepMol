import shap
import pandas as pd


# TODO: Add more features (different plots and methods)
class ShapValues(object):
    '''
    ...
    '''

    def __init__(self, dataset, model):
        '''
        :param dataset:
        :param model:
        '''

        self.dataset = dataset
        self.model = model
        self.shap_values = None

    # TODO: masker not working
    def computePermutationShap(self, masker=False, plot=True, **kwargs):
        columns_names = ['feat_' + str(i + 1) for i in range(self.dataset.X.shape[1])]
        X = pd.DataFrame(self.dataset.X, columns=columns_names)

        model = self.model.model

        if masker:
            y = self.dataset.y

            # build a clustering of the features based on shared information about y
            clustering = shap.utils.hclust(X, y)

            # above we implicitly used shap.maskers.Independent by passing a raw dataframe as the masker
            # now we explicitly use a Partition masker that uses the clustering we just computed
            masker = shap.maskers.Partition(X, clustering=clustering)

            # build a Permutation explainer and explain the model predictions on the given dataset
            explainer = shap.explainers.Permutation(model.predict_proba, masker)

        else:
            explainer = shap.explainers.Permutation(model.predict, X)

        self.shap_values = explainer(X)
        if plot:
            # visualize all the training set predictions
            if masker:
                shap.plots.bar(self.shap_values, **kwargs)
            else:
                shap.plots.beeswarm(self.shap_values, **kwargs)

    # TODO: masker not working
    # TODO: too much iterations needed (remove?)
    def computeExactShap(self, masker=False, plot=True, **kwargs):
        columns_names = ['feat_' + str(i + 1) for i in range(self.dataset.X.shape[1])]
        X = pd.DataFrame(self.dataset.X, columns=columns_names)

        model = self.model.model

        if masker:
            y = self.dataset.y

            # build a clustering of the features based on shared information about y
            clustering = shap.utils.hclust(X, y)

            # above we implicitly used shap.maskers.Independent by passing a raw dataframe as the masker
            # now we explicitly use a Partition masker that uses the clustering we just computed
            masker = shap.maskers.Partition(X, clustering=clustering)

            # build an Exact explainer and explain the model predictions on the given dataset
            explainer = shap.explainers.Exact(model.predict_proba, masker)
        else:
            explainer = shap.explainers.Exact(model.predict_proba, X)

        self.shap_values = explainer(X)
        if plot:
            # visualize all the training set predictions
            if masker:
                shap.plots.bar(self.shap_values, **kwargs)
            else:
                shap.plots.beeswarm(self.shap_values, **kwargs)

    # TODO: check why force is not working (maybe java plugin is missing?)
    def plotSampleExplanation(self, index=0, plot_type='waterfall'):
        if self.shap_values is None:
            print('Shap values not computed yet! Computing shap values...')
            self.computeShap(plot=False)

        if plot_type == 'waterfall':
            # visualize the nth prediction's explanation
            shap.plots.waterfall(self.shap_values[index])
        elif plot_type == 'force':
            shap.initjs()
            # visualize the first prediction's explanation with a force plot
            shap.plots.force(self.shap_values[index])
        else:
            raise ValueError('Plot type must be waterfall or force!')

    def plotFeatureExplanation(self, index='all'):
        if index == 'all':
            # summarize the effects of all the features
            shap.plots.beeswarm(self.shap_values)
        else:
            # create a dependence scatter plot to show the effect of a single feature across the whole dataset
            shap.plots.scatter(self.shap_values[:, index], color=self.shap_values)

    def plotHeatMap(self):
        if self.shap_values is not None:
            shap.plots.heatmap(self.shap_values)
        else:
            raise ValueError('Shap values not computed yet!')

    # TODO: check this again
    '''
    def plotPositiveClass(self):
        shap_values2 = self.shap_values[...,1]
        print(shap_values2)
        shap.plots.bar(shap_values2)

    def plotNegativeClass(self):
        shap_values2 = self.shap_values[...,0]
        shap.plots.bar(shap_values2)
    '''
