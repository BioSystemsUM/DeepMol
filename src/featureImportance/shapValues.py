import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.models.DeepChemModels import DeepChemModel
from src.Datasets.Datasets import NumpyDataset


# TODO: allow user to explain predictions for the test set as well

#TODO: Add more features (different plots and methods)
class ShapValues(object):
    '''
    ...
    '''

    def __init__(self, dataset, model, mode):
        '''
        :param dataset:
        :param model:
        '''

        self.model = model
        self.dataset = dataset
        self.mode = mode
        try:
            self.columns_names = ['feat_' + str(i + 1) for i in range(dataset.X.shape[1])]
        except:
            self.columns_names = None

        #X = pd.DataFrame(dataset.X, columns=self.columns_names)
        self.shap_values = None

    #TODO: masker not working
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
            else :
                shap.plots.beeswarm(self.shap_values, **kwargs)

    def computeDeepShap(self, n_background_samples=100, plot=True, **kwargs):
        # doesn't work for DeepChemModels (because of output shape)
        model = self.model.model.model
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # so that it works for Keras models with Batch Normalization
        background_inds = np.random.choice(self.dataset.X.shape[0], n_background_samples, replace=False)
        background = self.dataset.X[background_inds]
        explainer = shap.DeepExplainer(model, background)
        shap_vals = explainer.shap_values(self.dataset.X)
        expected_value = explainer.expected_value
        base_values = np.tile(expected_value, self.dataset.X.shape[0]) # this is necessary for waterfall plots
        self.shap_values = shap._explanation.Explanation(values=shap_vals[0], base_values=base_values,
                                                         data=self.dataset.X,
                                                         feature_names=self.columns_names)

        if plot:
            shap.plots.beeswarm(self.shap_values, **kwargs)

    def computeGradientShap(self, plot=True, **kwargs):
        # doesn't work for DeepChemModels (because of output shape)
        model = self.model.model.model
        explainer = shap.GradientExplainer(model, self.dataset.X)
        shap_vals = explainer.shap_values(self.dataset.X)

        expected_value = model.predict(self.dataset.X).mean(0) # TODO: change. i think this is only works for regression??
        # https://github.com/slundberg/shap/issues/1095
        base_values = np.tile(expected_value, self.dataset.X.shape[0]) # repeat value across all rows
        # base_values are necessary for waterfall plots
        self.shap_values = shap._explanation.Explanation(values=shap_vals[0], base_values=base_values,
                                                         data=self.dataset.X, feature_names=self.columns_names)

        if plot:
            shap.plots.beeswarm(self.shap_values, **kwargs)

    def computeKernelShap(self, n_background_samples=10, nsamples=100, plot=True, **kwargs):

        def f(data):
            new_dataset = NumpyDataset(mols=self.dataset.mols, X=data) # need to do this because KernelExplainer requires a function that only accepts numpy arrays or pandas DataFrames
            return self.model.predict(new_dataset)

        background = shap.sample(self.dataset.X, nsamples=n_background_samples)
        # background = shap.kmeans(self.dataset.X, n_background_samples)
        # background = self.dataset.X.median().values.reshape((1, self.dataset.X.shape[1]))
        if self.mode == 'classification':
            link = 'logit'
        else:
            link = 'identity'
        explainer = shap.KernelExplainer(f, background, link=link)
        shap_values = explainer.shap_values(self.dataset.X, nsamples=nsamples)
        print(shap_values)
        print(type(shap_values))

        if plot:
            shap.plots.beeswarm(self.shap_values, **kwargs)


    #TODO: check why force is not working (maybe java plugin is missing?)
    def plotSampleExplanation(self, index=0, plot_type='waterfall', save=False):
        if self.shap_values is None:
            print('Shap values not computed yet! Computing shap values...')
            self.computeShap(plot=False)

        if plot_type=='waterfall':
            # visualize the nth prediction's explanation
            shap.plots.waterfall(self.shap_values[index])
            if save:
                plt.tight_layout()
                plt.savefig('shap_sample_explanation_plot.png')
            else:
                plt.show()
        elif plot_type=='force':
            shap.initjs()
            # visualize the first prediction's explanation with a force plot
            shap.plots.force(self.shap_values[index])
        else:
            raise ValueError('Plot type must be waterfall or force!')

    def plotFeatureExplanation(self, index='all', save=False):
        if index=='all':
            # summarize the effects of all the features
            shap.plots.beeswarm(self.shap_values)
        else:
            # create a dependence scatter plot to show the effect of a single feature across the whole dataset
            shap.plots.scatter(self.shap_values[:, index], color=self.shap_values)

        if save:
            plt.tight_layout()
            plt.savefig('shap_feature_explanation_plot.png')
        else:
            plt.show()

    def plotHeatMap(self):
        if self.shap_values is not None:
            shap.plots.heatmap(self.shap_values)
        else:
            raise ValueError('Shap values not computed yet!')

    def plotBar(self, max_display=10, save=False):
        if self.shap_values is not None:
            shap.plots.bar(self.shap_values, max_display=max_display)
            if save:
                plt.tight_layout()
                plt.savefig('shap_bar_plot.png')
            else:
                plt.show()
        else:
            raise ValueError('Shap values not computed yet!')

    def plot_important_fp_bits(self, index, bit):
        # implement this here or just call the drawing functions from utils after we inspect the plots to
        # find the most important bits?
        pass

    def save_shap_values(self, output_filepath):
        df = pd.DataFrame(data=self.shap_values.values, columns=self.shap_values.feature_names)
        df.to_csv(output_filepath, index=False)


    #TODO: check this again
    '''
    def plotPositiveClass(self):
        shap_values2 = self.shap_values[...,1]
        print(shap_values2)
        shap.plots.bar(shap_values2)

    def plotNegativeClass(self):
        shap_values2 = self.shap_values[...,0]
        shap.plots.bar(shap_values2)
    '''
