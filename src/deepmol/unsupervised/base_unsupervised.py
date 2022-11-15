from abc import abstractmethod, ABC
from typing import Union

from deepmol.datasets import Dataset, NumpyDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from kneed import KneeLocator
import seaborn as sns

from sklearn import cluster, decomposition, manifold


# TODO: plot legends and labels are made for sweet vs non sweet --> change it to be general
class UnsupervisedLearn(ABC):
    """
    Class for unsupervised learning.

    A UnsupervisedLearn sampler receives a Dataset object and performs unsupervised learning.

    Subclasses need to implement a _unsupervised method to perform unsupervised learning.
    """

    def __init__(self):
        """
        Initialize the UnsupervisedLearn object.
        """
        if self.__class__ == UnsupervisedLearn:
            raise Exception('Abstract class UnsupervisedLearn should not be instantiated')
        self.features = None

    def runUnsupervised(self, dataset: Dataset, plot: bool = True):
        """
        Run unsupervised learning.

        Parameters
        ----------
        dataset: Dataset
            The dataset to perform unsupervised learning.
        plot: bool
            If True, plot the results of unsupervised learning.

        Returns
        -------
        x: NumpyDataset
            The dataset with the unsupervised features in dataset.X.
        """
        self.dataset = dataset
        self.features = dataset.X
        x = self._runUnsupervised(plot=plot)
        return x

    @abstractmethod
    def _runUnsupervised(self, plot: bool = True):
        """
        Run unsupervised learning.

        Parameters
        ----------
        plot: bool
            If True, plot the results of unsupervised learning.

        Returns
        -------
        x: NumpyDataset
            The dataset with the unsupervised features in dataset.X.
        """
        raise NotImplementedError


class PCA(UnsupervisedLearn):
    """
    Class to perform principal component analysis (PCA).

    Wrapper around scikit-learn PCA
    (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional
    space.
    """

    def __init__(self,
                 n_components: Union[int, float, str] = None,
                 copy: bool = True,
                 whiten: bool = False,
                 svd_solver: str = 'auto',
                 tol: float = 0.0,
                 iterated_power: Union[int, str] = 'auto',
                 random_state: int = None):
        """
        Parameters
        ----------
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
        """
        super().__init__()
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def _runUnsupervised(self, plot=True):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        plot: bool
            If True, plot the results of unsupervised learning.
        """
        pca = decomposition.PCA(n_components=self.n_components,
                                copy=self.copy,
                                whiten=self.whiten,
                                svd_solver=self.svd_solver,
                                tol=self.tol,
                                iterated_power=self.iterated_power,
                                random_state=self.random_state)
        if plot:
            self._plot()

        return NumpyDataset(mols=self.dataset.mols,
                            X=pca.fit_transform(self.features),
                            y=self.dataset.y,
                            ids=self.dataset.ids,
                            features2keep=self.dataset.features2keep,
                            n_tasks=self.dataset.n_tasks)

    def _plot(self):
        """
        Plot the results of unsupervised learning (PCA).
        """
        print('2 Components PCA: ')
        pca2comps = decomposition.PCA(n_components=2,
                                      copy=self.copy,
                                      whiten=self.whiten,
                                      svd_solver=self.svd_solver,
                                      tol=self.tol,
                                      iterated_power=self.iterated_power,
                                      random_state=self.random_state)
        components = pca2comps.fit_transform(self.features)

        dic = {0: "Not Sweet", 1: "Sweet"}
        colors_map = []
        for elem in self.dataset.y:
            colors_map.append(dic[elem])

        total_var = pca2comps.explained_variance_ratio_.sum() * 100

        fig = px.scatter(components, x=0, y=1, color=colors_map,
                         title=f'Total Explained Variance: {total_var:.2f}%',
                         labels={'0': 'PC 1', '1': 'PC 2'})
        fig.show()

        print('\n \n')
        print('3 Components PCA: ')
        pca3comps = decomposition.PCA(n_components=3,
                                      copy=self.copy,
                                      whiten=self.whiten,
                                      svd_solver=self.svd_solver,
                                      tol=self.tol,
                                      iterated_power=self.iterated_power,
                                      random_state=self.random_state)
        components = pca3comps.fit_transform(self.features)

        total_var = pca3comps.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(components,
                            x=0, y=1, z=2, color=colors_map,
                            title=f'Total Explained Variance: {total_var:.2f}%',
                            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
                            )
        fig.show()

        print('\n \n')

        if self.n_components is None:
            self.n_components = self.dataset.X.shape[1]
            print('%i Components PCA: ' % self.n_components)
        else:
            print('%i Components PCA: ' % self.n_components)

        pca_all = decomposition.PCA(n_components=self.n_components,
                                    copy=self.copy,
                                    whiten=self.whiten,
                                    svd_solver=self.svd_solver,
                                    tol=self.tol,
                                    iterated_power=self.iterated_power,
                                    random_state=self.random_state)
        components_all = pca_all.fit_transform(self.features)

        total_var = pca_all.explained_variance_ratio_.sum() * 100

        labels = {str(i): f"PC {i + 1}" for i in range(self.n_components)}

        fig = px.scatter_matrix(components_all,
                                color=colors_map,
                                dimensions=range(self.n_components),
                                labels=labels,
                                title=f'Total Explained Variance: {total_var:.2f}%',
                                )
        fig.update_traces(diagonal_visible=False)
        fig.show()

        print('\n \n')
        print('Explained Variance: ')
        pca_comp = decomposition.PCA()
        pca_comp.fit(self.features)
        exp_var_cumul = np.cumsum(pca_comp.explained_variance_ratio_)

        fig = px.area(x=range(1, exp_var_cumul.shape[0] + 1),
                      y=exp_var_cumul,
                      labels={"x": "# Components", "y": "Explained Variance"}
                      )

        fig.show()


class TSNE(UnsupervisedLearn):
    """
    Class to perform t-distributed Stochastic Neighbor Embedding (TSNE).

    Wrapper around scikit-learn TSNE
    (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)

    It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler
    divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.
    """

    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12.0,
                 learning_rate=200.0,
                 n_iter=1000,
                 n_iter_without_progress=300,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 init='random',
                 verbose=0,
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 n_jobs=None):
        """
        Parameters
        ----------
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
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs

    def _runUnsupervised(self, plot=True):
        """Fit X into an embedded space and return that transformed output."""
        X_embedded = manifold.TSNE(n_components=self.n_components,
                                   perplexity=self.perplexity,
                                   early_exaggeration=self.early_exaggeration,
                                   learning_rate=self.learning_rate,
                                   n_iter=self.n_iter,
                                   n_iter_without_progress=self.n_iter_without_progress,
                                   min_grad_norm=self.min_grad_norm,
                                   metric=self.metric,
                                   init=self.init,
                                   verbose=self.verbose,
                                   random_state=self.random_state,
                                   method=self.method,
                                   angle=self.angle,
                                   n_jobs=self.n_jobs)

        if plot:
            self._plot()

        return NumpyDataset(mols=self.dataset.mols,
                            X=X_embedded.fit_transform(self.features),
                            y=self.dataset.y,
                            ids=self.dataset.ids,
                            features2keep=self.dataset.features2keep,
                            n_tasks=self.dataset.n_tasks)

    def _plot(self):
        dic = {0: "Not Active (0)", 1: "Active (1)"}
        colors_map = []
        for elem in self.dataset.y:
            colors_map.append(dic[elem])

        print('2 Components t-SNE: ')
        tsne2comp = manifold.TSNE(n_components=2,
                                  perplexity=self.perplexity,
                                  early_exaggeration=self.early_exaggeration,
                                  learning_rate=self.learning_rate,
                                  n_iter=self.n_iter,
                                  n_iter_without_progress=self.n_iter_without_progress,
                                  min_grad_norm=self.min_grad_norm,
                                  metric=self.metric,
                                  init=self.init,
                                  verbose=self.verbose,
                                  random_state=self.random_state,
                                  method=self.method,
                                  angle=self.angle,
                                  n_jobs=self.n_jobs)

        projections2comp = tsne2comp.fit_transform(self.features)

        fig = px.scatter(projections2comp, x=0, y=1,
                         color=colors_map, labels={'color': 'Class'}
                         )
        fig.show()

        print('\n \n')
        print('3 Components t-SNE: ')
        tsne3comp = manifold.TSNE(n_components=3,
                                  perplexity=self.perplexity,
                                  early_exaggeration=self.early_exaggeration,
                                  learning_rate=self.learning_rate,
                                  n_iter=self.n_iter,
                                  n_iter_without_progress=self.n_iter_without_progress,
                                  min_grad_norm=self.min_grad_norm,
                                  metric=self.metric,
                                  init=self.init,
                                  verbose=self.verbose,
                                  random_state=self.random_state,
                                  method=self.method,
                                  angle=self.angle,
                                  n_jobs=self.n_jobs)

        projections3comp = tsne3comp.fit_transform(self.features)

        fig = px.scatter_3d(projections3comp, x=0, y=1, z=2,
                            color=colors_map, labels={'color': 'species'}
                            )
        fig.update_traces(marker_size=8)

        fig.show()


class KMeans(UnsupervisedLearn):
    """Class to perform K-Means clustering.

    Wrapper around scikit-learn K-Means.
    (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    """

    def __init__(self,
                 n_clusters='elbow',
                 init: str = 'k-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 0.0001,
                 verbose: int = 0,
                 random_state: int = None,
                 copy_x: bool = True,
                 algorithm: str = 'auto'):
        """
        Initialize KMeans object.

        Parameters
        ----------
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
        algorithm: str {“auto”, “full”, “elkan”}
            K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more
            efficient on data with well-defined clusters, by using the triangle inequality. However, it’s more memory
            intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _runUnsupervised(self, plot=True):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        plot: bool
            Whether to plot the results or not.

        Returns
        -------
        k_means.labels_: ndarray
            Index of the cluster each sample belongs to.
        """

        if self.n_clusters == 'elbow':
            self.n_clusters = self.elbow()

        k_means = cluster.KMeans(n_clusters=self.n_clusters,
                                 init=self.init,
                                 n_init=self.n_init,
                                 max_iter=self.max_iter,
                                 tol=self.tol,
                                 verbose=self.verbose,
                                 random_state=self.random_state,
                                 copy_x=self.copy_x,
                                 algorithm=self.algorithm)

        if plot:
            self._plot()

        k_means.fit_predict(self.features)
        return k_means.labels_

    def elbow(self):
        """
        Determine the optimal number of clusters using the elbow method.

        Returns
        -------
        int
            The optimal number of clusters.
        """
        wcss = []
        for i in range(1, 11):
            kmeans_elbow = cluster.KMeans(n_clusters=i,
                                          init=self.init,
                                          n_init=self.n_init,
                                          max_iter=self.max_iter,
                                          tol=self.tol,
                                          verbose=self.verbose,
                                          random_state=self.random_state,
                                          copy_x=self.copy_x,
                                          algorithm=self.algorithm)
            kmeans_elbow.fit(self.features)
            wcss.append(kmeans_elbow.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method Graph')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        clusters_df = pd.DataFrame({"cluster_errors": wcss, "num_clusters": range(1, 11)})
        elbow = KneeLocator(clusters_df.num_clusters.values,
                            clusters_df.cluster_errors.values,
                            S=1.0,
                            curve='convex',
                            direction='decreasing')

        print('Creating a K-means cluster with ' + str(elbow.knee) + ' clusters...')
        return elbow.knee

    def _plot(self):
        """
        Plot the results of the clustering.
        """
        # TODO: check the best approach to this problem
        if self.features.shape[1] > 11:
            print('Reduce the number of features to less than ten to get plot interpretability!')
        else:
            kmeans = cluster.KMeans(n_clusters=self.n_clusters,
                                    init=self.init,
                                    n_init=self.n_init,
                                    max_iter=self.max_iter,
                                    tol=self.tol,
                                    verbose=self.verbose,
                                    random_state=self.random_state,
                                    copy_x=self.copy_x,
                                    algorithm=self.algorithm)
            kmeans.fit(self.features)
            kmeans.predict(self.features)
            labels = kmeans.labels_

            labels = pd.DataFrame(labels)
            labels = labels.rename({0: 'labels'}, axis=1)
            ds = pd.concat((pd.DataFrame(self.features), labels), axis=1)
            sns.pairplot(ds, hue='labels')

            classes = pd.DataFrame(self.dataset.y)
            classes = classes.rename({0: 'classes'}, axis=1)
            ds = pd.concat((pd.DataFrame(self.features), classes), axis=1)
            sns.pairplot(ds, hue='classes')
