from Dataset.Dataset import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, decomposition, manifold


class UnsupervisedLearn(object):
    """Class for unsupervised learning.

    A UnsupervisedLearn sampler receives a Dataset object and performs unsupervised learning.

    Subclasses need to implement a _unsupervised method to perform unsupervised learning.
    """

    def __init__(self):
        if self.__class__ == UnsupervisedLearn:
            raise Exception('Abstract class UnsupervisedLearn should not be instantiated')

        self.features = None

    def runUnsupervised(self, dataset: Dataset, plot=True):

        self.features = dataset.features

        x = self._runUnsupervised(plot=plot)

        return x

    def plot(self):
        NotImplementedError("Each subclass must implement its own plot method.")


class PCA(UnsupervisedLearn):
    """Class to perform principal component analysis (PCA).

    Wrapper around scikit-learn PCA
    (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it
    to a lower dimensional space.
    """

    def __init__(self,
                 n_components=None,
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto',
                 random_state=None):
        """
        Parameters
        ----------
        n_components: int, float, None or str (default None)
            Number of components to keep. if n_components is not set all components are kept:

            n_components == min(n_samples, n_features)
            If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension.
            Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.

            If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount
            of variance that needs to be explained is greater than the percentage specified by n_components.

            If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features
            and n_samples.

            Hence, the None case results in: n_components == min(n_samples, n_features) - 1

        copy: bool, (default True)
            If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected
            results, use fit_transform(X) instead.

        whiten: bool, optional (default False)
            When True (False by default) the components_ vectors are multiplied by the square root of n_samples and
            then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.

            Whitening will remove some information from the transformed signal (the relative variance scales of the
            components) but can sometime improve the predictive accuracy of the downstream estimators by making their
            data respect some hard-wired assumptions.

        svd_solver: str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}, (default auto)
            If auto :
                The solver is selected by a default policy based on X.shape and n_components: if the input data is
                larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension
                of the data, then the more efficient ‘randomized’ method is enabled. Otherwise the exact full SVD is
                computed and optionally truncated afterwards.

            If full :
                run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components
                by postprocessing

            If arpack :
                run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires
                strictly 0 < n_components < min(X.shape)

            If randomized :
                run randomized SVD by the method of Halko et al.

        tol: float >= 0, optional (default 0.0)
            Tolerance for singular values computed by svd_solver == ‘arpack’.

        iterated_power: int >= 0, or ‘auto’, (default auto)
            Number of iterations for the power method computed by svd_solver == ‘randomized’.

        random_state: int, RandomState instance, (default None)
            Used when svd_solver == ‘arpack’ or ‘randomized’. Pass an int for reproducible results across
            multiple function calls.

        """
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state


    def _runUnsupervised(self, plot=True):
        """Fit the model with X and apply the dimensionality reduction on X."""
        pca = decomposition.PCA(n_components=self.n_components,
                                copy=self.copy,
                                whiten=self.whiten,
                                svd_solver=self.svd_solver,
                                tol=self.tol,
                                iterated_power=self.iterated_power,
                                random_state=self.random_state)

        pca.fit_transform(self.features)
        self.pca = pca
        print('asdasd')
        print('pca_components: ', self.pca.components_)
        print('explained_variance_ratio: ', self.pca.explained_variance_ratio_)

        if plot:
            self._plot()


    def _plot(self, colors="tomato", n_components=5):

        pca = self.pca

        print('????????????')
        print('!!!!!!!!!!!')
        xLabels = ["PC " + str(x + 1) for x in range(n_components)]
        yLabels = [x for x in range(0, 100, 5)]

        explica = pca.explained_variance_ratio_ * 100
        explica = pd.DataFrame(explica[:n_components], index=xLabels, columns=["Explained Variance"])

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))

        explica.plot(kind="bar", ax=ax[0], color="deepskyblue", zorder=3, edgecolor="teal")

        ax[0].grid(b=True, axis='y', zorder=0, color='lightcoral')

        ax[0].set_yticks(yLabels)
        ax[0].set_ylabel("%")

        print("2 Principal Components explain", np.cumsum(pca.explained_variance_ratio_), "% of the data variance.")

        print(pca)
        # plt.figure(figsize=(20,8))
        ax[1].scatter(pca[:, 0], pca[:, 1], marker='.', c=colors)

        mean_X = np.mean(pca[:, 0])
        std_X = np.std(pca[:, 0])
        max_X = mean_X + 3 * std_X
        min_X = mean_X - 3 * std_X
        if np.max(pca[:, 0]) < max_X: max_X = np.max(pca[:, 0])
        if np.min(pca[:, 0]) > min_X: min_X = np.min(pca[:, 0])

        mean_Y = np.mean(pca[:, 1])
        std_Y = np.std(pca[:, 1])
        max_Y = mean_Y + 3 * std_Y
        min_Y = mean_Y - 3 * std_Y
        if np.max(pca[:, 1]) < max_Y: max_Y = np.max(pca[:, 1])
        if np.min(pca[:, 1]) > min_Y: min_Y = np.min(pca[:, 1])

        plt.xlim([min_X, max_X])
        plt.ylim([min_Y, max_Y])

        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        '''


class TSNE(UnsupervisedLearn):
    """Class to perform t-distributed Stochastic Neighbor Embedding (TSNE).

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


    def _runUnsupervised(self):
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
        return X_embedded.fit_transform(self.features)

    # TODO: implement
    def plot(self):
        pass


class KMeans(UnsupervisedLearn):
    """Class to perform K-Means clustering.

    Wrapper around scikit-learn K-Means.
    (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    """

    def __init__(self,
                 n_clusters=8,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 tol=0.0001,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 algorithm='auto'):
        """
        Parameters
        ----------
        n_clusters: int, default=8
            The number of clusters to form as well as the number of centroids to generate.

        init: {‘k-means++’, ‘random’, ndarray, callable}, default=’k-means++’
            Method for initialization:
                ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up
                convergence. See section Notes in k_init for more details.

                ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.

                If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

                If a callable is passed, it should take arguments X, n_clusters and a random state and return an
                initialization.

        n_init: int, default=10
            Number of time the k-means algorithm will be run with different centroid seeds. The final results will be
            the best output of n_init consecutive runs in terms of inertia.

        max_iter: int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.

        tol: float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two
            consecutive iterations to declare convergence.

        verbose: int, default=0
            Verbosity mode.

        random_state: int, RandomState instance, default=None
            Determines random number generation for centroid initialization. Use an int to make the randomness
            deterministic.

        copy_x: bool, default=True
            When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True
            (default), then the original data is not modified. If False, the original data is modified, and put back
            before the function returns, but small numerical differences may be introduced by subtracting and then
            adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if
            copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x
            is False.

        algorithm: {“auto”, “full”, “elkan”}, default=”auto”
            K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more
            efficient on data with well-defined clusters, by using the triangle inequality. However it’s more memory
            intensive due to the allocation of an extra array of shape (n_samples, n_clusters).

        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm


    def _runUnsupervised(self):
        """Compute cluster centers and predict cluster index for each sample."""
        k_means = cluster.KMeans(n_clusters=self.n_clusters,
                                 init=self.init,
                                 n_init=self.n_init,
                                 max_iter=self.max_iter,
                                 tol=self.tol,
                                 verbose=self.verbose,
                                 random_state=self.random_state,
                                 copy_x=self.copy_x,
                                 algorithm=self.algorithm)
        return k_means.fit_predict(self.features)

    # TODO: implement
    def plot(self):
        pass