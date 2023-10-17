from abc import abstractmethod, ABC
from typing import Tuple

from deepmol.base import Transformer
from deepmol.datasets import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from kneed import KneeLocator

from sklearn import cluster, decomposition, manifold

from deepmol.loggers.logger import Logger
from deepmol.utils.decorators import modify_object_inplace_decorator


class UnsupervisedLearn(ABC, Transformer):
    """
    Class for unsupervised learning.

    A UnsupervisedLearn sampler receives a Dataset object and performs unsupervised learning.

    Subclasses need to implement a _unsupervised method to perform unsupervised learning.
    """

    def __init__(self):
        """
        Initialize the UnsupervisedLearn object.
        """
        self.logger = Logger()
        super().__init__()

    @modify_object_inplace_decorator
    def run(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Run unsupervised learning.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.
        kwargs:
            Additional arguments to pass to the _run_unsupervised method.

        Returns
        -------
        df: Dataset
            The dataset with the unsupervised features in dataset.X.
        """
        dataset._X, dataset.feature_names = self._run_unsupervised(dataset=dataset, **kwargs)
        return dataset

    @abstractmethod
    def _run_unsupervised(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Run unsupervised learning.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.
        kwargs:
            Additional arguments to pass to the _unsupervised method.

        Returns
        -------
        x: Dataset
            The dataset with the unsupervised features in dataset.X.
        """

    @abstractmethod
    def plot(self, x_new: np.ndarray, path: str = None, **kwargs) -> None:
        """
        Plot the results of unsupervised learning.

        Parameters
        ----------
        x_new: np.ndarray
            Transformed values.
        path: str
            The path to save the plot.
        **kwargs:
            Additional arguments to pass to the plot function.
        """


class PCA(UnsupervisedLearn):
    """
    Class to perform principal component analysis (PCA).

    Wrapper around scikit-learn PCA
    (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional
    space.
    """

    def __init__(self, **kwargs) -> None:
        """
        Parameters
        ----------
        kwargs:
            Additional arguments to pass to the sklearn.decomposition.PCA class including:
            n_components: Union[int, float, str]
                Number of components to keep. if n_components is not set all components are kept:
                If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension.
                Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.
                If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount
                of variance that needs to be explained is greater than the percentage specified by n_components.
                If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features
                and n_samples.
            copy: bool
                If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected
                results, use fit_transform(X) instead.
            whiten: bool
                When True the components_ vectors are multiplied by the square root of n_samples and then divided by the
                singular values to ensure uncorrelated outputs with unit component-wise variances.
            svd_solver: str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
                If auto :
                    The solver is selected by a default policy based on X.shape and n_components: if the input data is
                    larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension
                    of the data, then the more efficient ‘randomized’ method is enabled. Otherwise, the exact full SVD is
                    computed and optionally truncated afterwards.
                If full :
                    run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components
                    by postprocessing
                If arpack :
                    run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires
                    strictly 0 < n_components < min(X.shape)
                If randomized :
                    run randomized SVD by the method of Halko et al.
            tol: float
                Tolerance for singular values computed by svd_solver == ‘arpack’.
            iterated_power: Union[int, str]
                Number of iterations for the power method computed by svd_solver == ‘randomized’. 'auto' selects it
                automatically.
            random_state: int
                Used when svd_solver == ‘arpack’ or ‘randomized’. Pass an int for reproducible results across multiple
                function calls.
            n_oversamples: int
                Additional number of random vectors to sample the range of M to ensure proper conditioning.
                Only used by randomized SVD solver when svd_solver == 'randomized'.
            power_iteration_normalizer: str
                Power iteration normalizer for randomized SVD solver. Available options are ‘auto’, ‘QR’, ‘LU’, ‘none’.
        """
        super().__init__()
        self.pca = decomposition.PCA(**kwargs)

    def _run_unsupervised(self, dataset: Dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.
        kwargs:
            Additional arguments to pass to the _unsupervised method.

        Returns
        -------
        x_new: np.ndarray
            Transformed values.
        feature_names: np.ndarray
            The names of the features.
        """
        self.dataset = dataset
        x_new = self.pca.fit_transform(dataset.X)
        feature_names = np.array([f'PCA_{i}' for i in range(x_new.shape[1])])
        return x_new, feature_names

    def plot(self, x_new: np.ndarray, path: str = None, **kwargs) -> None:
        """
        Plot the results of unsupervised learning (PCA).

        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        path: str
            Path to save the plot.
        **kwargs:
            Additional arguments to pass to the plot method.
        """
        self.logger.info(f'{x_new.shape[1]} Components PCA: ')

        total_var = self.pca.explained_variance_ratio_.sum() * 100

        if self.dataset.mode == 'classification':
            y = [str(i) for i in self.dataset.y]
        else:
            y = self.dataset.y

        if x_new.shape[1] == 2:
            fig = px.scatter(x_new, x=0, y=1, color=y,
                             title=f'Total Explained Variance: {total_var:.2f}%',
                             labels={'0': 'PC 1', '1': 'PC 2', 'color': self.dataset.label_names[0]}, **kwargs)
        elif x_new.shape[1] == 3:
            fig = px.scatter_3d(x_new, x=0, y=1, z=2, color=y,
                                title=f'Total Explained Variance: {total_var:.2f}%',
                                labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': self.dataset.label_names[0]})
        else:
            labels = {str(i): f"PC {i + 1}" for i in range(x_new.shape[1])}
            labels['color'] = self.dataset.label_names[0]
            fig = px.scatter_matrix(x_new,
                                    color=y,
                                    dimensions=range(x_new.shape[1]),
                                    labels=labels,
                                    title=f'Total Explained Variance: {total_var:.2f}%',
                                    **kwargs)
            fig.update_traces(diagonal_visible=False)
        fig.show()
        if path is not None:
            fig.write_image(path)

    def plot_explained_variance(self, path: str = None, **kwargs) -> None:
        """
        Plot the explained variance.

        Parameters
        ----------
        path: str
            Path to save the plot.
        **kwargs:
            Additional arguments to pass to the plot method.
        """
        self.logger.info('Explained Variance: ')
        exp_var_cumul = np.cumsum(self.pca.explained_variance_ratio_)
        fig = px.area(x=range(1, exp_var_cumul.shape[0] + 1),
                      y=exp_var_cumul,
                      labels={"x": "# Components", "y": "Explained Variance"},
                      **kwargs)
        fig.show()
        if path is not None:
            fig.write_image(path)

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fit the model with dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        self: PCA
            The fitted model.
        """
        self.dataset = dataset
        self.pca.fit(dataset.X)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Apply dimensionality reduction on dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        dataset._X = self.pca.transform(dataset.X)
        dataset.feature_names = np.array([f'PCA_{i}' for i in range(dataset.X.shape[1])])
        return dataset


class TSNE(UnsupervisedLearn):
    """
    Class to perform t-distributed Stochastic Neighbor Embedding (TSNE).

    Wrapper around scikit-learn TSNE
    (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)

    It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler
    divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.
    """

    def __init__(self, **kwargs) -> None:
        """
        Parameters
        ----------
        kwargs:
            Additional arguments to pass to the sklearn.manifold.TSNE class including:
            n_components: int, optional (default: 2)
                Dimension of the embedded space.

            perplexity: float, optional (default: 30)
                The perplexity is related to the number of nearest neighbors that is used in other manifold learning
                algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5
                and 50. Different values can result in significanlty different results.

            early_exaggeration: float, optional (default: 12.0)
                Controls how tight natural clusters in the original space are in the embedded space and how much space
                will be between them. For larger values, the space between natural clusters will be larger in the embedded
                space. Again, the choice of this parameter is not very critical. If the cost function increases during
                initial optimization, the early exaggeration factor or the learning rate might be too high.

            learning_rate: float, optional (default: 200.0)
                The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the
                data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the
                learning rate is too low, most points may look compressed in a dense cloud with few outliers. If the cost
                function gets stuck in a bad local minimum increasing the learning rate may help.

            n_iter: int, optional (default: 1000)
                Maximum number of iterations for the optimization. Should be at least 250.

            n_iter_without_progress: int, optional (default: 300)
                Maximum number of iterations without progress before we abort the optimization, used after 250 initial
                iterations with early exaggeration. Note that progress is only checked every 50 iterations so this value
                is rounded to the next multiple of 50.

            min_grad_norm: float, optional (default: 1e-7)
                If the gradient norm is below this threshold, the optimization will be stopped.

            metric: string or callable, optional
                The metric to use when calculating distance between instances in a feature array. If metric is a string,
                it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter, or a metric
                listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is “precomputed”, X is assumed to be a distance
                matrix. Alternatively, if metric is a callable function, it is called on each pair of instances (rows) and
                the resulting value recorded. The callable should take two arrays from X as input and return a value
                indicating the distance between them. The default is “euclidean” which is interpreted as squared euclidean
                distance.

            init: string or numpy array, optional (default: “random”)
                Initialization of embedding. Possible options are ‘random’, ‘pca’, and a numpy array of shape
                (n_samples, n_components). PCA initialization cannot be used with precomputed distances and is usually more
                globally stable than random initialization.

            verbose: int, optional (default: 0)
                Verbosity level.

            random_state: int, RandomState instance, default=None
                Determines the random number generator. Pass an int for reproducible results across multiple function calls.
                Note that different initializations might result in different local minima of the cost function.

            method: string (default: ‘barnes_hut’)
                By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time.
                method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be
                used when nearest-neighbor errors need to be better than 3%. However, the exact method cannot scale to
                millions of examples.

            angle: float (default: 0.5)
                Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
                ‘angle’ is the angular size (referred to as theta in [3]) of a distant node as measured from a point. If
                this size is below ‘angle’ then it is used as a summary node of all points contained within it. This method
                is not very sensitive to changes in this parameter in the range of 0.2 - 0.8. Angle less than 0.2 has
                quickly increasing computation time and angle greater 0.8 has quickly increasing error.

            n_jobs: int or None, optional (default=None)
                The number of parallel jobs to run for neighbors search. This parameter has no impact when
                metric="precomputed" or (metric="euclidean" and method="exact"). None means 1 unless in a
                joblib.parallel_backend context. -1 means using all processors.

        """
        super().__init__()
        self.tsne = manifold.TSNE(**kwargs)

    def _run_unsupervised(self, dataset: Dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be transformed.

        Returns
        -------
        x_new: np.ndarray
            The transformed output.
        feature_names: np.ndarray
            The feature names.
        """
        self.dataset = dataset
        x_new = self.tsne.fit_transform(dataset.X)
        feature_names = np.array([f"tsne_{i}" for i in range(x_new.shape[1])])
        return x_new, feature_names

    def plot(self, x_new: np.ndarray, path: str = None, **kwargs) -> None:
        self.logger.info(f'{x_new.shape[1]} Components t-SNE: ')

        if self.dataset.mode == 'classification':
            y = [str(i) for i in self.dataset.y]
        else:
            y = self.dataset.y

        if x_new.shape[1] == 2:
            fig = px.scatter(x_new, x=0, y=1, color=y, labels={'color': self.dataset.label_names[0]}, **kwargs)
        elif x_new.shape[1] == 3:
            fig = px.scatter_3d(x_new, x=0, y=1, z=2, color=y, labels={'color': self.dataset.label_names[0]}, **kwargs)
        else:
            fig = px.scatter_matrix(x_new,
                                    color=y,
                                    dimensions=range(x_new.shape[1]),
                                    labels={'color': self.dataset.label_names[0]},
                                    **kwargs)
            fig.update_traces(diagonal_visible=False)
        fig.show()
        if path:
            fig.write_image(path)

    def _fit(self, dataset: Dataset) -> 'TSNE':
        """
        Fit the model with dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        self: TSNE
            The fitted model.
        """
        self.dataset = dataset
        self.tsne.fit(dataset.X)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Apply dimensionality reduction on dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        dataset._X = self.tsne.fit_transform(dataset.X)
        dataset.feature_names = np.array([f"tsne_{i}" for i in range(dataset.X.shape[1])])
        return dataset


class KMeans(UnsupervisedLearn):
    """Class to perform K-Means clustering.

    Wrapper around scikit-learn K-Means.
    (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize KMeans object.

        Parameters
        ----------
        kwargs:
            Keyword arguments to pass to scikit-learn K-Means including:
            n_clusters: Union[int, str]
                The number of clusters to form as well as the number of centroids to generate.
                'elbow' uses the elbow method to determine the most suited number of clusters.
            init: str {‘k-means++’, ‘random’, ndarray, callable}
                Method for initialization:
                    ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up
                    convergence. See section Notes in k_init for more details.
                    ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
                    If a ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
                    If a callable is passed, it should take arguments X, n_clusters and a random state and return an
                    initialization.
            n_init: int
                Number of time the k-means algorithm will be run with different centroid seeds. The final results will be
                the best output of n_init consecutive runs in terms of inertia.
            max_iter: int
                Maximum number of iterations of the k-means algorithm for a single run.
            tol: float
                Relative tolerance in regard to Frobenius norm of the difference in the cluster centers of two
                consecutive iterations to declare convergence.
            verbose: int
                Verbosity mode.
            random_state: int
                Determines random number generation for centroid initialization. Use an int to make the randomness
                deterministic.
            copy_x: bool
                When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True
                (default), then the original data is not modified. If False, the original data is modified, and put back
                before the function returns, but small numerical differences may be introduced by subtracting and then
                adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if
                copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x
                is False.
            algorithm: str {"lloyd", "elkan", "auto", "full"}
                K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`. The “elkan” variation is more
                efficient on data with well-defined clusters, by using the triangle inequality. However, it’s more memory
                intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
        """
        super().__init__()
        self.k_means = None
        self.kwargs = kwargs

    def _get_kmeans_instance(self, dataset: Dataset, **kwargs) -> None:
        """
        Return the KMeans instance.

        Parameters
        ----------
        dataset: Dataset
            Dataset to cluster.
        kwargs:
            Additional keyword arguments to pass to the elbow method.
        """
        if 'n_clusters' not in self.kwargs or self.kwargs['n_clusters'] == 'elbow':
            self.kwargs['n_clusters'] = 'elbow'
            self.logger.info('Using elbow method to determine number of clusters.')
            n_clusters = self._elbow(dataset, **kwargs)
            self.kwargs['n_clusters'] = n_clusters

        self.k_means = cluster.KMeans(**self.kwargs)

    def _run_unsupervised(self, dataset: Dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        dataset: Dataset
            Dataset to cluster.
        kwargs:
            Additional keyword arguments to pass to the elbow method.

        Returns
        -------
        x_new: np.ndarray
            The transformed output.
        feature_names: np.ndarray
            The feature names.
        """
        self.dataset = dataset

        self._get_kmeans_instance(dataset, **kwargs)

        x_new = self.k_means.fit_transform(dataset.X)
        feature_names = np.array([f"cluster_{i}" for i in range(x_new.shape[1])])
        return x_new, feature_names

    def _elbow(self, dataset: Dataset, **kwargs):
        """
        Determine the optimal number of clusters using the elbow method.

        Parameters
        ----------
        dataset: Dataset
            Dataset to cluster.
        kwargs:
            Additional keyword arguments to pass to the elbow method.
            kwargs include:
                path: str
                    Path to save the elbow method graph. By default, the graph is not saved.
                S: float
                    The sensitivity of the elbow method. By default, S = 0.1.
                curve: str
                    If 'concave', algorithm will detect knees. If 'convex', it will detect elbows.
                    By default, curve = 'concave'.
                direction: str
                    One of {"increasing", "decreasing"}. By default, direction = 'increasing'.
                interp_method: str
                    One of {"interp1d", "polynomial"}. By default, interp_method = 'interp1d'.
                online: bool
                    kneed will correct old knee points if True, will return first knee if False. By default False.
                polynomial_degree: int
                    The degree of the fitting polynomial. Only used when interp_method="polynomial".
                    This argument is passed to numpy polyfit `deg` parameter. By default 7.

        Returns
        -------
        int
            The optimal number of clusters.
        """
        # kwargs without n_clusters
        k_means_kwargs = self.kwargs.copy()
        k_means_kwargs.pop('n_clusters')
        wcss = []
        for i in range(1, 11):
            kmeans_elbow = cluster.KMeans(n_clusters=i,
                                          **k_means_kwargs)
            kmeans_elbow.fit(dataset.X)
            wcss.append(kmeans_elbow.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method Graph')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        if 'path' in kwargs:
            plt.savefig(kwargs['path'])
            kwargs.pop('path')

        clusters_df = pd.DataFrame({"cluster_errors": wcss, "num_clusters": range(1, 11)})
        elbow = KneeLocator(clusters_df.num_clusters.values,
                            clusters_df.cluster_errors.values,
                            **kwargs)

        self.logger.info(f'The optimal number of clusters is {elbow.knee} as determined by the elbow method.')
        return elbow.knee

    def plot(self, x_new: np.ndarray, path: str = None, **kwargs) -> None:
        """
        Plot the results of the clustering.

        Parameters
        ----------
        x_new: np.ndarray
            Transformed dataset.
        path: str
            Path to save the plot.
        **kwargs:
            Additional arguments for the plot.
        """
        self.logger.info('Plotting the results of the clustering.')
        if x_new.shape[1] == 2:
            fig = px.scatter(x_new, x=0, y=1, color=[str(kl) for kl in self.k_means.labels_],
                             labels={'color': 'cluster'}, **kwargs)
        elif x_new.shape[1] == 3:
            fig = px.scatter_3d(x_new, x=0, y=1, z=2, color=[str(kl) for kl in self.k_means.labels_],
                                labels={'color': 'cluster'}, **kwargs)
        else:
            fig = px.scatter_matrix(x_new, color=[str(kl) for kl in self.k_means.labels_],
                                    dimensions=range(x_new.shape[1]), labels={'color': 'cluster'}, **kwargs)
            fig.update_traces(diagonal_visible=False)
        fig.show()
        if path:
            fig.write_image(path)

    def _fit(self, dataset: Dataset) -> 'KMeans':
        """
        Fit the model with dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        self: KMeans
            The fitted model.
        """
        self.dataset = dataset
        # Using fit does not allow to pass additional arguments to the elbow method
        self._get_kmeans_instance(dataset)
        self.k_means.fit(dataset.X)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Apply dimensionality reduction on dataset.X.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        dataset._X = self.k_means.transform(dataset.X)
        dataset.feature_names = np.array([f"cluster_{i}" for i in range(dataset.X.shape[1])])
        return dataset
