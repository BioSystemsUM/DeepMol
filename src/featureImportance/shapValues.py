import os
import pickle
import copy
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.Datasets.Datasets import NumpyDataset


#TODO: Add more features (different plots and methods)
class ShapValues(object):
    '''
    ...
    '''

    def __init__(self, dataset, model, mode):
        '''
        :param dataset: Dataset to explain (can be the dataset the model was originally trained on or a test dataset)
        :param model: Model to explain
        '''

        self.model = model
        self.dataset = dataset # TODO: need to see if this affects
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
                shap.plots.bar(copy.deepcopy(self.shap_values), **kwargs)
            else:
                shap.plots.beeswarm(copy.deepcopy(self.shap_values), **kwargs) # # using a copy because beeswarm modifies the shap values for some reason

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
                shap.plots.bar(copy.deepcopy(self.shap_values), **kwargs)
            else :
                shap.plots.beeswarm(copy.deepcopy(self.shap_values), **kwargs) # using a copy because beeswarm modifies the shap values for some reason

    def computeDeepShap(self, train_dataset, n_background_samples=100, plot=True, **kwargs):
        # train_dataset is the dataset that the model was trained on
        # doesn't work for DeepChemModels (because of output shape)
        # TODO: see this example to see if I can get this working with TextCNN
        model = self.model.model.model
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # so that it works for Keras models with Batch Normalization
        background_inds = np.random.choice(train_dataset.X.shape[0], n_background_samples, replace=False)
        background = train_dataset.X[background_inds] # background always taken from the training set
        explainer = shap.DeepExplainer(model, background)
        shap_vals = explainer.shap_values(self.dataset.X)
        expected_value = explainer.expected_value
        base_values = np.tile(expected_value, self.dataset.X.shape[0]) # this is necessary for waterfall plots
        self.shap_values = shap._explanation.Explanation(values=shap_vals[0], base_values=base_values,
                                                         data=self.dataset.X,
                                                         feature_names=self.columns_names)

        if plot:
            shap.plots.beeswarm(copy.deepcopy(self.shap_values), **kwargs) # using a copy because beeswarm modifies shap_values for some reason

    def computeGradientShap(self, train_dataset, plot=True, **kwargs):
        # doesn't work for DeepChemModels (because of output shape)
        model = self.model.model.model
        explainer = shap.GradientExplainer(model, train_dataset.X) # uses train_dataset as the background

        expected_value = model.predict(train_dataset.X).mean(0)
        base_values = np.tile(expected_value, self.dataset.X.shape[0]) # repeat value across all rows
        # base_values are necessary for waterfall plots
        shap_vals = explainer.shap_values(self.dataset.X)
        self.shap_values = shap._explanation.Explanation(values=shap_vals[0], base_values=base_values,
                                                         data=self.dataset.X, feature_names=self.columns_names)

        if plot:
            shap.plots.beeswarm(copy.deepcopy(self.shap_values), **kwargs)

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

        if plot:
            shap.plots.beeswarm(copy.deepcopy(self.shap_values), **kwargs)

    #TODO: check why force is not working (maybe java plugin is missing?)
    def plotSampleExplanation(self, index=0, plot_type='waterfall', save=False, output_dir=None, max_display=20):
        if self.shap_values is None:
            print('Shap values not computed yet! Computing shap values...')
            self.computeShap(plot=False)

        shap_values = copy.deepcopy(self.shap_values) # because some plotting functions like beeswarm may be modifiying the explanation object

        if plot_type=='waterfall':
            # visualize the nth prediction's explanation
            shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
            if save:
                if output_dir is not None:
                    output_path = os.path.join(output_dir, 'shap_sample%s_explanation_plot.png' % str(index))
                else:
                    output_path = 'shap_sample%s_explanation_plot.png' % str(index)
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()
        elif plot_type=='force':
            shap.initjs()
            # visualize the first prediction's explanation with a force plot
            shap.plots.force(shap_values[index])
        else:
            raise ValueError('Plot type must be waterfall or force!')

    def plotFeatureExplanation(self, index='all', save=False, output_dir=None, max_display=20):
        shap_values = copy.deepcopy(self.shap_values) # because some plotting functions like beeswarm may be modifiying the explanation object

        if index=='all':
            # summarize the effects of all the features
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        else:
            # create a dependence scatter plot to show the effect of a single feature across the whole dataset
            shap.plots.scatter(shap_values[:, index], color=self.shap_values)

        if save:
            if output_dir is not None:
                output_path = os.path.join(output_dir, 'shap_feature_explanation_plot.png')
            else:
                output_path = 'shap_feature_explanation_plot.png'
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plotHeatMap(self):
        if self.shap_values is not None:
            shap.plots.heatmap(copy.deepcopy(self.shap_values))
        else:
            raise ValueError('Shap values not computed yet!')

    def plotBar(self, max_display=20, save=False, output_dir=None):
        if self.shap_values is not None:
            shap.plots.bar(copy.deepcopy(self.shap_values), max_display=max_display, show=False)
            if save:
                if output_dir is not None:
                    output_path = os.path.join(output_dir, 'shap_bar_plot.png')
                else:
                    output_path = 'shap_bar_plot.png'
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()
        else:
            raise ValueError('Shap values not computed yet!')

    def save_shap_values(self, output_filepath):
        df = pd.DataFrame(data=self.shap_values.values, columns=self.shap_values.feature_names)
        df.to_csv(output_filepath, index=False)

    def save_explanation_object(self, output_filepath):
        with open(output_filepath, 'wb') as f:
            pickle.dump(self.shap_values, f)

    def load_explanation_object(self, filepath):
        with open(filepath, 'rb') as f:
            self.shap_values = pickle.load(f)

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
